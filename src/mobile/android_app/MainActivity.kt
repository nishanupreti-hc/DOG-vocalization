package com.dogspeak.translator

import android.Manifest
import android.content.pm.PackageManager
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.util.Log
import android.view.MotionEvent
import android.view.View
import android.widget.*
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import kotlinx.coroutines.*
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import kotlin.math.*

class MainActivity : AppCompatActivity() {
    
    private lateinit var recordButton: ImageButton
    private lateinit var statusText: TextView
    private lateinit var resultCard: LinearLayout
    private lateinit var intentText: TextView
    private lateinit var confidenceText: TextView
    private lateinit var explanationText: TextView
    private lateinit var adviceText: TextView
    private lateinit var waveformView: WaveformView
    
    private var audioRecord: AudioRecord? = null
    private var isRecording = false
    private var recordingJob: Job? = null
    
    // Audio configuration
    private val sampleRate = 16000
    private val channelConfig = AudioFormat.CHANNEL_IN_MONO
    private val audioFormat = AudioFormat.ENCODING_PCM_16BIT
    private val bufferSize = AudioRecord.getMinBufferSize(sampleRate, channelConfig, audioFormat)
    
    // ML Model
    private var tfliteInterpreter: Interpreter? = null
    private val modelInputSize = 64 * 126 // 64 mel bins, ~4 seconds
    
    // Intent labels (matching Python taxonomy)
    private val tier1Labels = arrayOf(
        "Alarm/Guard", "Territorial", "Play Invitation", "Distress/Separation",
        "Pain/Discomfort", "Attention Seeking", "Whine/Appeal", "Growl/Threat",
        "Growl/Play", "Howl/Contact", "Yip/Puppy", "Other/Unknown"
    )
    
    companion object {
        private const val REQUEST_RECORD_AUDIO_PERMISSION = 200
        private const val TAG = "DogSpeak"
    }
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        
        initializeViews()
        setupRecordButton()
        loadTFLiteModel()
        requestAudioPermission()
    }
    
    private fun initializeViews() {
        recordButton = findViewById(R.id.recordButton)
        statusText = findViewById(R.id.statusText)
        resultCard = findViewById(R.id.resultCard)
        intentText = findViewById(R.id.intentText)
        confidenceText = findViewById(R.id.confidenceText)
        explanationText = findViewById(R.id.explanationText)
        adviceText = findViewById(R.id.adviceText)
        waveformView = findViewById(R.id.waveformView)
        
        // Initially hide results
        resultCard.visibility = View.GONE
        statusText.text = "Tap and hold to record your dog"
    }
    
    private fun setupRecordButton() {
        recordButton.setOnTouchListener { _, event ->
            when (event.action) {
                MotionEvent.ACTION_DOWN -> {
                    if (hasAudioPermission()) {
                        startRecording()
                    } else {
                        requestAudioPermission()
                    }
                    true
                }
                MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> {
                    if (isRecording) {
                        stopRecording()
                    }
                    true
                }
                else -> false
            }
        }
    }
    
    private fun loadTFLiteModel() {
        try {
            val modelFile = assets.openFd("dogspeak_model.tflite")
            val inputStream = FileInputStream(modelFile.fileDescriptor)
            val fileChannel = inputStream.channel
            val startOffset = modelFile.startOffset
            val declaredLength = modelFile.declaredLength
            val modelBuffer = fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
            
            tfliteInterpreter = Interpreter(modelBuffer)
            Log.d(TAG, "TFLite model loaded successfully")
        } catch (e: Exception) {
            Log.e(TAG, "Error loading TFLite model", e)
            Toast.makeText(this, "Model loading failed", Toast.LENGTH_SHORT).show()
        }
    }
    
    private fun hasAudioPermission(): Boolean {
        return ContextCompat.checkSelfPermission(
            this, Manifest.permission.RECORD_AUDIO
        ) == PackageManager.PERMISSION_GRANTED
    }
    
    private fun requestAudioPermission() {
        ActivityCompat.requestPermissions(
            this,
            arrayOf(Manifest.permission.RECORD_AUDIO),
            REQUEST_RECORD_AUDIO_PERMISSION
        )
    }
    
    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        
        if (requestCode == REQUEST_RECORD_AUDIO_PERMISSION) {
            if (grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                Toast.makeText(this, "Audio permission granted", Toast.LENGTH_SHORT).show()
            } else {
                Toast.makeText(this, "Audio permission required", Toast.LENGTH_SHORT).show()
            }
        }
    }
    
    private fun startRecording() {
        if (isRecording) return
        
        try {
            audioRecord = AudioRecord(
                MediaRecorder.AudioSource.MIC,
                sampleRate,
                channelConfig,
                audioFormat,
                bufferSize
            )
            
            audioRecord?.startRecording()
            isRecording = true
            
            // Update UI
            recordButton.setImageResource(R.drawable.ic_stop)
            recordButton.setBackgroundResource(R.drawable.record_button_active)
            statusText.text = "Recording... Release to analyze"
            resultCard.visibility = View.GONE
            
            // Start recording coroutine
            recordingJob = CoroutineScope(Dispatchers.IO).launch {
                recordAudio()
            }
            
        } catch (e: Exception) {
            Log.e(TAG, "Error starting recording", e)
            Toast.makeText(this, "Recording failed", Toast.LENGTH_SHORT).show()
        }
    }
    
    private fun stopRecording() {
        if (!isRecording) return
        
        isRecording = false
        recordingJob?.cancel()
        
        audioRecord?.stop()
        audioRecord?.release()
        audioRecord = null
        
        // Update UI
        recordButton.setImageResource(R.drawable.ic_mic)
        recordButton.setBackgroundResource(R.drawable.record_button_inactive)
        statusText.text = "Processing..."
    }
    
    private suspend fun recordAudio() {
        val audioBuffer = ShortArray(bufferSize)
        val audioData = mutableListOf<Short>()
        
        while (isRecording) {
            val readResult = audioRecord?.read(audioBuffer, 0, bufferSize) ?: 0
            
            if (readResult > 0) {
                // Add to audio data
                audioData.addAll(audioBuffer.take(readResult))
                
                // Update waveform on UI thread
                withContext(Dispatchers.Main) {
                    waveformView.updateWaveform(audioBuffer.take(readResult).toFloatArray())
                }
            }
        }
        
        // Process recorded audio
        if (audioData.isNotEmpty()) {
            processAudio(audioData.toShortArray())
        }
    }
    
    private suspend fun processAudio(audioData: ShortArray) {
        withContext(Dispatchers.IO) {
            try {
                // Convert to float and normalize
                val floatAudio = audioData.map { it.toFloat() / Short.MAX_VALUE }.toFloatArray()
                
                // Extract log-mel features (simplified version)
                val logMelFeatures = extractLogMelFeatures(floatAudio)
                
                // Run inference
                val prediction = runInference(logMelFeatures)
                
                // Generate explanation
                val explanation = generateExplanation(prediction)
                
                // Update UI on main thread
                withContext(Dispatchers.Main) {
                    displayResults(prediction, explanation)
                }
                
            } catch (e: Exception) {
                Log.e(TAG, "Error processing audio", e)
                withContext(Dispatchers.Main) {
                    statusText.text = "Processing failed. Try again."
                }
            }
        }
    }
    
    private fun extractLogMelFeatures(audio: FloatArray): FloatArray {
        // Simplified log-mel extraction for demo
        // In production, use proper STFT and mel filterbank
        
        val targetLength = 4 * sampleRate // 4 seconds
        val processedAudio = if (audio.size < targetLength) {
            audio + FloatArray(targetLength - audio.size) { 0f }
        } else {
            audio.take(targetLength).toFloatArray()
        }
        
        // Mock log-mel features (64 mels x 126 frames)
        val features = FloatArray(modelInputSize)
        
        // Simple spectral analysis
        val frameSize = 1024
        val hopSize = 320
        val numFrames = min(126, (processedAudio.size - frameSize) / hopSize)
        
        for (frame in 0 until numFrames) {
            val start = frame * hopSize
            val frameData = processedAudio.sliceArray(start until min(start + frameSize, processedAudio.size))
            
            // Simple energy calculation per mel bin
            for (mel in 0 until 64) {
                val freqStart = (mel * frameData.size / 64)
                val freqEnd = ((mel + 1) * frameData.size / 64)
                
                var energy = 0f
                for (i in freqStart until min(freqEnd, frameData.size)) {
                    energy += frameData[i] * frameData[i]
                }
                
                features[frame * 64 + mel] = ln(max(energy, 1e-10f))
            }
        }
        
        return features
    }
    
    private fun runInference(features: FloatArray): DogSpeakPrediction {
        val interpreter = tfliteInterpreter ?: return createMockPrediction()
        
        try {
            // Prepare input
            val inputBuffer = ByteBuffer.allocateDirect(features.size * 4)
            inputBuffer.order(ByteOrder.nativeOrder())
            features.forEach { inputBuffer.putFloat(it) }
            
            // Prepare outputs
            val tier1Output = Array(1) { FloatArray(tier1Labels.size) }
            val tier2Output = Array(1) { FloatArray(16) } // 16 tier2 classes
            val confidenceOutput = Array(1) { FloatArray(1) }
            
            val outputs = mapOf(
                0 to tier1Output,
                1 to tier2Output,
                2 to confidenceOutput
            )
            
            // Run inference
            interpreter.runForMultipleInputsOutputs(arrayOf(inputBuffer), outputs)
            
            // Process results
            val tier1Probs = tier1Output[0]
            val tier1Index = tier1Probs.indices.maxByOrNull { tier1Probs[it] } ?: 0
            val tier1Confidence = tier1Probs[tier1Index]
            
            return DogSpeakPrediction(
                tier1Intent = tier1Labels[tier1Index],
                tier1Confidence = tier1Confidence,
                tier2Tags = listOf(), // Simplified for demo
                overallConfidence = confidenceOutput[0][0]
            )
            
        } catch (e: Exception) {
            Log.e(TAG, "Inference error", e)
            return createMockPrediction()
        }
    }
    
    private fun createMockPrediction(): DogSpeakPrediction {
        // Mock prediction for demo when model is not available
        val mockIntents = arrayOf("Play Invitation", "Attention Seeking", "Alarm/Guard")
        val randomIntent = mockIntents.random()
        
        return DogSpeakPrediction(
            tier1Intent = randomIntent,
            tier1Confidence = 0.75f + (Math.random() * 0.2f).toFloat(),
            tier2Tags = listOf("indoor", "high_energy"),
            overallConfidence = 0.8f
        )
    }
    
    private fun generateExplanation(prediction: DogSpeakPrediction): DogSpeakExplanation {
        // Simplified explanation generation
        val explanations = mapOf(
            "Play Invitation" to "Your dog is excited and wants to play with you!",
            "Attention Seeking" to "Your dog is trying to get your attention for something.",
            "Alarm/Guard" to "Your dog is alerting you to something they noticed.",
            "Distress/Separation" to "Your dog seems anxious or worried about something."
        )
        
        val advice = mapOf(
            "Play Invitation" to "Engage in interactive play or exercise to channel their energy positively.",
            "Attention Seeking" to "Check if they need something basic like food, water, or bathroom time.",
            "Alarm/Guard" to "Acknowledge their alert and check what caught their attention.",
            "Distress/Separation" to "Provide comfort and reassurance, check for any obvious stressors."
        )
        
        return DogSpeakExplanation(
            explanation = explanations[prediction.tier1Intent] ?: "Your dog is trying to communicate something.",
            advice = advice[prediction.tier1Intent] ?: "Observe their body language for more context."
        )
    }
    
    private fun displayResults(prediction: DogSpeakPrediction, explanation: DogSpeakExplanation) {
        intentText.text = prediction.tier1Intent
        confidenceText.text = "${(prediction.tier1Confidence * 100).toInt()}% confident"
        explanationText.text = explanation.explanation
        adviceText.text = explanation.advice
        
        resultCard.visibility = View.VISIBLE
        statusText.text = "Tap and hold to record again"
        
        // Animate result card
        resultCard.alpha = 0f
        resultCard.animate()
            .alpha(1f)
            .setDuration(300)
            .start()
    }
    
    override fun onDestroy() {
        super.onDestroy()
        tfliteInterpreter?.close()
        audioRecord?.release()
    }
}

data class DogSpeakPrediction(
    val tier1Intent: String,
    val tier1Confidence: Float,
    val tier2Tags: List<String>,
    val overallConfidence: Float
)

data class DogSpeakExplanation(
    val explanation: String,
    val advice: String
)
