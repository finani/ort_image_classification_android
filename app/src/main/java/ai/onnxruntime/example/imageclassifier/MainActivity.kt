package ai.onnxruntime.example.imageclassifier

import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtLoggingLevel
import ai.onnxruntime.OrtSession
import ai.onnxruntime.OrtSession.SessionOptions
import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.drawable.BitmapDrawable
import android.os.Bundle
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import kotlinx.android.synthetic.main.activity_main.*
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {
    private val backgroundExecutor: ExecutorService by lazy { Executors.newSingleThreadExecutor() }
    private val labelData: List<String> by lazy { readLabels() }

    private var imageCapture: ImageCapture? = null
    private var imageAnalysis: ImageAnalysis? = null
    private var enableNNAPI: Boolean = false
    private var ortEnv: OrtEnvironment? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        // Request Camera permission
        if (allPermissionsGranted()) {
            startCamera()

//            val ortSession: OrtSession? = CreateOrtSession()
//            repeat(10) {
//                val imgBitmap = loadImageFromFile()
//                val rawBitmap = imgBitmap?.let { Bitmap.createScaledBitmap(it, 224, 224, false) }
//    //            val bitmap = rawBitmap?.rotate(image.imageInfo.rotationDegrees.toFloat())
//                val bitmap = rawBitmap
//
//                if (bitmap != null) {
//                    val imgData = preprocess(bitmap)
//                    val inputName = ortSession?.inputNames?.iterator()?.next()
//                    var result = Result()
//                    val shape = longArrayOf(1, 224, 224, 3)
//                    val ortEnv = OrtEnvironment.getEnvironment()
//                    ortEnv.use {
//                        // Create input tensor
//                        val inputTensor = OnnxTensor.createTensor(ortEnv, imgData, shape)
//                        val startTime = SystemClock.uptimeMillis()
//                        inputTensor.use {
//                            // Run the inference and get the output tensor
//                            val output =
//                                ortSession?.run(Collections.singletonMap(inputName, inputTensor))
//                            output.use {
//                                // Populate the result
//                                result.processTimeMs = SystemClock.uptimeMillis() - startTime
//                                @Suppress("UNCHECKED_CAST")
//                                val labelVals = ((output?.get(0)?.value) as Array<FloatArray>)[0]
//                                result.detectedIndices = argMax(labelVals)
//                                for (idx in result.detectedIndices) {
//                                    result.detectedScore.add(labelVals[idx])
//                                }
//                                output.close()
//                            }
//                        }
//                    }
//                    print(result)
//                    print(result.toString())
//                }
//            }

        } else {
            ActivityCompat.requestPermissions(
                    this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS)
        }

        enable_nnapi_toggle.setOnCheckedChangeListener { _, isChecked ->
            enableNNAPI = isChecked
            imageAnalysis?.clearAnalyzer()
            val ortAnalyzer = ORTAnalyzer(CreateOrtSession(), ::updateUI)
            imageAnalysis?.setAnalyzer(backgroundExecutor, ORTAnalyzer(CreateOrtSession(), ::updateUI))
            val path = ""
            val drawable = getDrawable(R.drawable.bts)
            val bitmapDrawable = drawable as BitmapDrawable
            val bitmap = bitmapDrawable.bitmap
            ortAnalyzer.analyzeBts(bitmap)
        }
    }

    private fun startCamera() {
        // Initialize ortEnv
        ortEnv = OrtEnvironment.getEnvironment(OrtLoggingLevel.ORT_LOGGING_LEVEL_FATAL)

        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
//
        cameraProviderFuture.addListener(Runnable {
//            val cameraProvider: ProcessCameraProvider = cameraProviderFuture.get()
//
//            // Preview
//            val preview = Preview.Builder()
//                    .setTargetAspectRatio(AspectRatio.RATIO_16_9)
//                    .build()
//                    .also {
//                        it.setSurfaceProvider(viewFinder.surfaceProvider)
//                    }
//
//            imageCapture = ImageCapture.Builder()
//                    .setTargetAspectRatio(AspectRatio.RATIO_16_9)
//                    .build()
//
//            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

            imageAnalysis = ImageAnalysis.Builder()
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .build()
                    .also {
                        it.setAnalyzer(backgroundExecutor, ORTAnalyzer(CreateOrtSession(), ::updateUI))
                    }

//            try {
//                cameraProvider.unbindAll()
//
//                cameraProvider.bindToLifecycle(
//                        this, cameraSelector, preview, imageCapture, imageAnalysis)
//            } catch (exc: Exception) {
//                Log.e(TAG, "Use case binding failed", exc)
//            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(baseContext, it) == PackageManager.PERMISSION_GRANTED
    }

    override fun onDestroy() {
        super.onDestroy()
        backgroundExecutor.shutdown()
        ortEnv?.close()
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
        if (requestCode == REQUEST_CODE_PERMISSIONS) {
            if (allPermissionsGranted()) {
//                startCamera()
            } else {
                Toast.makeText(this,
                        "Permissions not granted by the user.",
                        Toast.LENGTH_SHORT).show()
                finish()
            }

        }
    }

    private fun updateUI(result: Result) {
        if (result.detectedScore.isEmpty())
            return

        runOnUiThread {
            percentMeter.progress = (result.detectedScore[0] * 100).toInt()
            detected_item_1.text = labelData[result.detectedIndices[0]]
            detected_item_value_1.text = "%.2f%%".format(result.detectedScore[0] * 100)

            if (result.detectedIndices.size > 1) {
                detected_item_2.text = labelData[result.detectedIndices[1]]
                detected_item_value_2.text = "%.2f%%".format(result.detectedScore[1] * 100)
            }

            if (result.detectedIndices.size > 2) {
                detected_item_3.text = labelData[result.detectedIndices[2]]
                detected_item_value_3.text = "%.2f%%".format(result.detectedScore[2] * 100)
            }

            inference_time_value.text = result.processTimeMs.toString() + "ms"
        }
    }

    private fun readModel(): ByteArray {
        return resources.openRawResource(R.raw.mobilenet_v1_float).readBytes();
    }

    private fun readLabels(): List<String> {
        return resources.openRawResource(R.raw.labels).bufferedReader().readLines()
    }

    private fun CreateOrtSession(): OrtSession? {
        val so = SessionOptions()
        so.use {
            // Set to use 2 intraOp threads for CPU EP
            so.setIntraOpNumThreads(2)

            if (enableNNAPI)
                so.addNnapi()

            return ortEnv?.createSession(readModel(), so)
        }
    }

    fun loadImageFromFile(): Bitmap? {
        try {
            return BitmapFactory.decodeFile("ai/onnxruntime/example/imageclassifier/bts.png")
        } catch (e: Exception) {
            e.printStackTrace()
        }
        return null
    }

    fun argMax(labelVals: FloatArray): List<Int> {
        var indices = mutableListOf<Int>()
        for (k in 0..2) {
            var max: Float = 0.0f
            var idx: Int = 0
            for (i in 0..labelVals.size - 1) {
                val label_val = labelVals[i]
                if (label_val > max && !indices.contains(i)) {
                    max = label_val
                    idx = i
                }
            }

            indices.add(idx)
        }

        return indices.toList()
    }

    companion object {
        public const val TAG = "ORTImageClassifier"
        private const val REQUEST_CODE_PERMISSIONS = 10
        private val REQUIRED_PERMISSIONS = arrayOf(Manifest.permission.CAMERA)
    }
}
