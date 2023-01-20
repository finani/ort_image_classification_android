package ai.onnxruntime.example.imageclassifier

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.graphics.Bitmap
import android.graphics.Matrix
import android.graphics.RectF
import android.os.SystemClock
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import java.util.*
import kotlin.collections.ArrayList
import kotlin.math.exp
import kotlin.math.max


internal data class Result(
        var detectedIndices: List<Int> = emptyList(),
        var detectedScore: MutableList<Float> = mutableListOf<Float>(),
        var processTimeMs: Long = 0
) {}

internal class ORTAnalyzer(
        private val ortSession: OrtSession?,
        private val uiUpdateCallBack: (Result) -> Unit
) : ImageAnalysis.Analyzer {

    // Get index of top 3 values
    private fun argMax(labelVals: FloatArray): List<Int> {
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

    private fun softMax(modelResult: FloatArray): FloatArray {
        var labelVals = modelResult.copyOf()
        val max = labelVals.max()
        var sum = 0.0f

        // Get the reduced sum
        for (i in labelVals.indices) {
            labelVals[i] = exp(labelVals[i] - max!!)
            sum += labelVals[i]
        }

        if (sum != 0.0f) {
            for (i in labelVals.indices) {
                labelVals[i] /= sum
            }
        }

        return labelVals
    }

    // Rotate the image of the input bitmap
    fun Bitmap.rotate(degrees: Float): Bitmap? {
        val matrix = Matrix().apply { postRotate(degrees) }
        return Bitmap.createBitmap(this, 0, 0, width, height, matrix, true)
    }

    override fun analyze(image: ImageProxy) {
        // Convert the input image to bitmap and resize to 224x224 for model input
        val imgBitmap = image.toBitmap()
        val rawBitmap = imgBitmap?.let { Bitmap.createScaledBitmap(it, 224, 224, false) }
        val bitmap = rawBitmap?.rotate(image.imageInfo.rotationDegrees.toFloat())

        if (bitmap != null) {
            val imgData = preprocess(bitmap)
            val inputName = ortSession?.inputNames?.iterator()?.next()
            var result = Result()
            val shape = longArrayOf(1, 224, 224, 3)
            val ortEnv = OrtEnvironment.getEnvironment()
            ortEnv.use {
                // Create input tensor
                val input_tensor = OnnxTensor.createTensor(ortEnv, imgData, shape)
                val startTime = SystemClock.uptimeMillis()
                input_tensor.use {
                    // Run the inference and get the output tensor
                    val output = ortSession?.run(Collections.singletonMap(inputName, input_tensor))
                    output.use {
                        // Populate the result
                        result.processTimeMs = SystemClock.uptimeMillis() - startTime
                        @Suppress("UNCHECKED_CAST")
                        val labelVals = ((output?.get(0)?.value) as Array<FloatArray>)[0]
                        result.detectedIndices = argMax(labelVals)
                        for (idx in result.detectedIndices) {
                            result.detectedScore.add(labelVals[idx])
                        }
                        output.close()
                    }
                }
            }

            // Update the UI
            uiUpdateCallBack(result)
        }

        image.close()
    }

    fun analyzeBts(aBitmap: Bitmap) {
        // Convert the input image to bitmap and resize to 224x224 for model input
//        val rawBitmap = Bitmap.createScaledBitmap(aBitmap, 320, 192, false)
//        val rawBitmap = Bitmap.createScaledBitmap(aBitmap, 224, 224, false)
//        val bitmap = rawBitmap?.rotate(image.imageInfo.rotationDegrees.toFloat())
        val bitmap = aBitmap

        if (bitmap != null) {
            val imgData = preprocess(bitmap)
            val inputName = ortSession?.inputNames?.iterator()?.next()
            var result = Result()
//            val shape = longArrayOf(1, 224, 224, 3) // mobilenet
            val shape = longArrayOf(1, 3, 192, 320) // yolov7
//            val shape = longArrayOf(1, 3, 1024, 1024) // ppyolo (rotate)
            val ortEnv = OrtEnvironment.getEnvironment()
            ortEnv.use {
                // Create input tensor
                val input_tensor = OnnxTensor.createTensor(ortEnv, imgData, shape)
                val input1 = input_tensor.floatBuffer
                val input2 = input1.array()
                val input2Size = input2.size
                val startTime = SystemClock.uptimeMillis()
                input_tensor.use {
                    // Run the inference and get the output tensor
                    val output = ortSession?.run(Collections.singletonMap(inputName, input_tensor))
                    output.use {
                        // Populate the result
                        result.processTimeMs = SystemClock.uptimeMillis() - startTime
                        val temp = output?.get("output")?.get()
                        val temp2 = temp as OnnxTensor
                        val temp3 = temp2.floatBuffer
                        val temp4 = temp3.array()
                        val temp5 = temp4.size

//                        val iii = 0
//                        val ttt0 = temp2.floatBuffer[iii]
//                        val ttt1 = temp2.floatBuffer[iii + 1]
//                        val ttt2 = temp2.floatBuffer[iii + 2]
//                        val ttt3 = temp2.floatBuffer[iii + 3]
//                        val ttt4 = temp2.floatBuffer[iii + 4]
//                        val ttt5 = temp2.floatBuffer[iii + 5]
//                        val ttt6 = temp2.floatBuffer[iii + 6]
//                        val ttt7 = temp2.floatBuffer[iii + 7]

                        val nmsInput: ArrayList<nmsResult> = arrayListOf()
                        val objectCount = output?.size() ?: 0
                        val reshapedOutput: ArrayList<ArrayList<Float>> = arrayListOf()
                        for (idxx in 0 until objectCount) {
                            output?.get(idxx).let { result ->
                                val resultFloatArray = (result as OnnxTensor).floatBuffer.array()
                                if (resultFloatArray.size > 7) {
                                    for (idx in 0 until temp5 / 8) {
                                        reshapedOutput.add(arrayListOf(
                                            resultFloatArray[idx*8 + 0],
                                            resultFloatArray[idx*8 + 1],
                                            resultFloatArray[idx*8 + 2],
                                            resultFloatArray[idx*8 + 3],
                                            resultFloatArray[idx*8 + 4],
                                            resultFloatArray[idx*8 + 5],
                                            resultFloatArray[idx*8 + 6],
                                            resultFloatArray[idx*8 + 7]
                                        ))
                                        if (resultFloatArray[idx*8 + 4] < 0.25) {
                                            continue
                                        }
                                        val tempRect = RectF(
//                                            resultFloatArray[idx*8 + 0] - resultFloatArray[idx*8 + 2] / 2.0F,
//                                            resultFloatArray[idx*8 + 1] - resultFloatArray[idx*8 + 3] / 2.0F,
//                                            resultFloatArray[idx*8 + 0] + resultFloatArray[idx*8 + 2] / 2.0F,
//                                            resultFloatArray[idx*8 + 1] + resultFloatArray[idx*8 + 3] / 2.0F
                                            resultFloatArray[idx*8 + 0],
                                            resultFloatArray[idx*8 + 1],
                                            resultFloatArray[idx*8 + 2],
                                            resultFloatArray[idx*8 + 3]
                                        )
                                        var classIdx = 0
                                        var maxConfidence = resultFloatArray[idx*8 + 5]
                                        if (resultFloatArray[idx*8 + 6] > maxConfidence) {
                                            classIdx = 1
                                            maxConfidence = resultFloatArray[idx*8 + 6]
                                        }
                                        if (resultFloatArray[idx*8 + 7] > maxConfidence) {
                                            classIdx = 2
                                            maxConfidence = resultFloatArray[idx*8 + 7]
                                        }
                                        // 4번은 0.25 이하 버리고 시작
                                        // bbox, 5,6,7max, index

                                        val tempResult = nmsResult(
                                            idx,
                                            resultFloatArray[idx*8 + 4],
                                            classIdx,
                                            resultFloatArray[idx*8 + 4] * maxConfidence,
                                            tempRect)
                                        nmsInput.add(tempResult)
                                    }
                                }
                            }
                        }

                        val nmsOutput = nonMaxSuppression(nmsInput, 10, 0.2F)

                        println(nmsOutput?.size)

//                        val labelVals = ((output?.get(0)?.value) as Array<FloatArray>)[0]
//                        result.detectedIndices = argMax(labelVals)
//                        for (idx in result.detectedIndices) {
//                            result.detectedScore.add(labelVals[idx])
//                        }
                        output.close()
                    }
                }
            }

            // Update the UI
            uiUpdateCallBack(result)
        }
    }

    internal data class nmsResult(val rawIndex: Int, val confidence: Float, val classIndex: Int, val score: Float, val rect: RectF) {}

    fun nonMaxSuppression(
        boxes: ArrayList<nmsResult>,
        limit: Int,
        threshold: Float
    ): ArrayList<nmsResult>? {

        // Do an argsort on the confidence scores, from high to low.
//        Collections.sort(boxes,
//            object : Comparator<nmsResult?>() {
//                fun compare(o1: nmsResult, o2: nmsResult): Int {
//                    return o1.score.compareTo(o2.score)
//                }
//            })
        boxes.sortWith(Comparator { a, b -> b.score.compareTo(a.score) })
        val selected: ArrayList<nmsResult> = ArrayList()
        val active = BooleanArray(boxes.size)
        Arrays.fill(active, true)
        var numActive = active.size

        // The algorithm is simple: Start with the box that has the highest score.
        // Remove any remaining boxes that overlap it more than the given threshold
        // amount. If there are any boxes left (i.e. these did not overlap with any
        // previous boxes), then repeat this procedure, until no more boxes remain
        // or the limit has been reached.
        var done = false
        var i = 0
        while (i < boxes.size && !done) {
            if (active[i]) {
                val boxA = boxes[i]
                selected.add(boxA)
                if (selected.size >= limit) break
                for (j in i + 1 until boxes.size) {
                    if (active[j]) {
                        val boxB = boxes[j]
                        if (IOU(boxA.rect, boxB.rect) > threshold) {
                            active[j] = false
                            numActive -= 1
                            if (numActive <= 0) {
                                done = true
                                break
                            }
                        }
                    }
                }
            }
            i++
        }
        return selected
    }

    fun IOU(a: RectF, b: RectF): Float {
        val areaA: Float = (a.right - a.left) * (a.bottom - a.top)
        if (areaA <= 0.0) return 0.0f
        val areaB: Float = (b.right - b.left) * (b.bottom - b.top)
        if (areaB <= 0.0) return 0.0f
        val intersectionMinX: Float = Math.max(a.left, b.left)
        val intersectionMinY: Float = Math.max(a.top, b.top)
        val intersectionMaxX: Float = Math.min(a.right, b.right)
        val intersectionMaxY: Float = Math.min(a.bottom, b.bottom)
        val intersectionArea = Math.max(intersectionMaxY - intersectionMinY, 0f) *
                Math.max(intersectionMaxX - intersectionMinX, 0f)
        return intersectionArea / (areaA + areaB - intersectionArea)
    }

    // We can switch analyzer in the app, need to make sure the native resources are freed
    protected fun finalize() {
        ortSession?.close()
    }
}