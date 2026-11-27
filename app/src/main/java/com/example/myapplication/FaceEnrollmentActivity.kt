package com.example.myapplication

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.ImageFormat
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.Rect
import android.graphics.YuvImage
import android.os.Bundle
import android.util.Log
import android.widget.Button
import android.widget.ProgressBar
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.annotation.OptIn
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ExperimentalGetImage
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.Face
import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetectorOptions
import com.google.mlkit.vision.face.FaceLandmark
import java.io.ByteArrayOutputStream
import java.util.Locale
import java.util.concurrent.Executors
import kotlin.math.abs
import kotlin.math.sqrt

@OptIn(ExperimentalGetImage::class)
class FaceEnrollmentActivity : AppCompatActivity() {

    companion object {
        private const val TAG = "FACEVERIFY"

        private const val MIN_FACE_BRIGHTNESS = 95.0
        private const val MAX_DARK_PIXEL_RATIO = 0.20
        private const val MAX_BRIGHT_PIXEL_RATIO = 0.20

        private const val DARK_PIXEL_LIMIT = 55.0
        private const val BRIGHT_PIXEL_LIMIT = 245.0

        private const val REQUIRED_STABLE_FRAMES = 8
    }

    private data class IlluminationResult(
        val averageBrightness: Double,
        val darkPixelRatio: Double,
        val brightPixelRatio: Double
    ) {
        val isAdequate: Boolean
            get() =
                averageBrightness >= MIN_FACE_BRIGHTNESS &&
                        darkPixelRatio <= MAX_DARK_PIXEL_RATIO &&
                        brightPixelRatio <= MAX_BRIGHT_PIXEL_RATIO
    }

    private lateinit var previewView: PreviewView
    private lateinit var progress: ProgressBar
    private lateinit var progressText: TextView
    private lateinit var poseText: TextView
    private lateinit var statusText: TextView
    private lateinit var finishButton: Button

    private val cameraExecutor = Executors.newSingleThreadExecutor()
    private val embeddingExecutor = Executors.newSingleThreadExecutor()

    private var cameraProvider: ProcessCameraProvider? = null

    @Volatile
    private var processingFrame = false

    @Volatile
    private var transitioningEnrollment = false

    private var currentPose = 0
    private var stableFrames = 0
    private var noGlassesMode = false

    @Volatile
    private var isCapturingEmbedding = false

    @Volatile
    private var illuminationAdequate = false

    private val capturedEmbeddings = mutableListOf<FloatArray>()
    private val capturedMlKitEmbeddings = mutableListOf<FloatArray>()
    private val capturedAlignedBitmaps = mutableListOf<Bitmap>()

    private var embeddingProcessor: FaceEmbeddingProcessor? = null
    private var mediaPipeDetector: MediaPipeFaceDetector? = null

    private val detector by lazy {
        val options = FaceDetectorOptions.Builder()
            .setPerformanceMode(
                FaceDetectorOptions.PERFORMANCE_MODE_FAST
            )
            .setLandmarkMode(
                FaceDetectorOptions.LANDMARK_MODE_ALL
            )
            .enableTracking()
            .build()

        FaceDetection.getClient(options)
    }

    private val cameraPermissionLauncher =
        registerForActivityResult(
            ActivityResultContracts.RequestPermission()
        ) { granted ->
            if (granted) {
                startCamera()
            } else {
                statusText.text =
                    "Necesitamos permiso para utilizar la cámara."
            }
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_face_enrollment)

        noGlassesMode = intent.getBooleanExtra(
            "noGlassesMode",
            false
        )

        previewView = findViewById(R.id.previewView)
        progress = findViewById(R.id.progressEnrollment)
        progressText = findViewById(R.id.txtEnrollmentProgress)
        poseText = findViewById(R.id.txtCurrentPose)
        statusText = findViewById(R.id.txtCameraStatus)
        finishButton = findViewById(R.id.btnFinishEnrollment)

        progress.max = 3
        updateInterface()

        try {
            embeddingProcessor = FaceEmbeddingProcessor(this)
            mediaPipeDetector = MediaPipeFaceDetector(this)

            Log.d(TAG, "Motores biométricos preparados")

            ModelDiagnostics.inspect(
                this,
                "MediaPipeFaceDetector.tflite"
            )

            ModelDiagnostics.inspect(
                this,
                "MediaPipeFaceLandmarkDetector.tflite"
            )

        } catch (e: Exception) {
            Log.e(TAG, "Error inicializando motores", e)

            statusText.text =
                "Error inicializando motores: ${e.localizedMessage}"

            finishButton.isEnabled = false
            return
        }

        finishButton.setOnClickListener {
            saveEnrollment()
        }

        checkCameraPermission()
    }

    // =========================================================
    // GUARDADO DEL REGISTRO FACIAL
    // =========================================================

    private fun saveEnrollment() {
        if (
            currentPose != 3 ||
            capturedEmbeddings.size != 3 ||
            capturedMlKitEmbeddings.size != 3
        ) {
            statusText.text = "Debes completar las tres poses."
            return
        }

        val username = intent.getStringExtra("username")

        if (username.isNullOrBlank()) {
            statusText.text = "No se identificó el usuario."
            return
        }

        finishButton.isEnabled = false

        statusText.text = if (noGlassesMode) {
            "Guardando registro complementario..."
        } else {
            "Guardando registro facial..."
        }

        val front = capturedEmbeddings[0].copyOf()
        val left = capturedEmbeddings[1].copyOf()
        val right = capturedEmbeddings[2].copyOf()

        try {
            embeddingExecutor.execute {

                val saved = try {
                    val database = UserDatabase(this)

                    try {
                        if (noGlassesMode) {
                            database.saveNoGlassesFaceEmbeddings(
                                username,
                                front,
                                left,
                                right
                            )
                        } else {
                            database.saveFaceEmbeddings(
                                username,
                                front,
                                left,
                                right
                            )
                        }
                    } finally {
                        database.close()
                    }

                } catch (e: Exception) {
                    Log.e(
                        TAG,
                        "Error guardando plantillas faciales",
                        e
                    )
                    false
                }

                runOnUiThread {
                    if (isFinishing || isDestroyed) {
                        return@runOnUiThread
                    }

                    if (!saved) {
                        statusText.text =
                            "No se pudieron guardar las plantillas."

                        finishButton.isEnabled = true
                        return@runOnUiThread
                    }

                    // REGISTRO PRINCIPAL
                    if (!noGlassesMode) {

                        transitioningEnrollment = true

                        Log.d(
                            TAG,
                            "Registro principal guardado para $username"
                        )

                        android.app.AlertDialog.Builder(this)
                            .setTitle("¡Registro facial completado!")
                            .setMessage(
                                "Tu registro principal se ha completado correctamente.\n\n" +
                                        "¿Quieres mejorar la flexibilidad de FaceVerify " +
                                        "registrando una plantilla facial complementaria?\n\n" +
                                        "Este registro es opcional, pero muy recomendable. " +
                                        "Si utilizas gafas u otros accesorios que puedan " +
                                        "modificar tu apariencia, retíralos para realizar " +
                                        "el registro adicional. Así contaremos con una " +
                                        "referencia complementaria de tu rostro para " +
                                        "facilitar la verificación cuando tu apariencia cambie."
                            )
                            .setPositiveButton(
                                "Sí, realizar registro adicional"
                            ) { _, _ ->
                                beginAdditionalEnrollment()
                            }
                            .setNegativeButton(
                                "Ahora no"
                            ) { _, _ ->
                                Toast.makeText(
                                    this,
                                    "Registro completado. Ya puedes iniciar sesión.",
                                    Toast.LENGTH_LONG
                                ).show()

                                goToLogin()
                            }
                            .setCancelable(false)
                            .show()

                    } else {

                        // REGISTRO COMPLEMENTARIO COMPLETADO

                        Log.d(
                            TAG,
                            "Registro complementario guardado para $username"
                        )

                        transitioningEnrollment = true

                        Toast.makeText(
                            this,
                            "¡Registro complementario completado!",
                            Toast.LENGTH_LONG
                        ).show()

                        goToLogin()
                    }
                }
            }

        } catch (e: Exception) {
            Log.e(
                TAG,
                "No se pudo iniciar el guardado",
                e
            )

            statusText.text =
                "No se pudo iniciar el guardado de plantillas."

            finishButton.isEnabled = true
        }
    }

    // =========================================================
    // INICIO DEL REGISTRO COMPLEMENTARIO
    // =========================================================

    private fun beginAdditionalEnrollment() {

        Log.d(
            TAG,
            "Reiniciando captura para registro complementario."
        )

        noGlassesMode = true

        transitioningEnrollment = false

        currentPose = 0
        stableFrames = 0

        illuminationAdequate = false
        isCapturingEmbedding = false

        // No modificamos processingFrame manualmente.
        // El analizador controla ese estado al terminar cada frame.

        capturedEmbeddings.clear()
        capturedMlKitEmbeddings.clear()

        capturedAlignedBitmaps.forEach {
            if (!it.isRecycled) {
                it.recycle()
            }
        }

        capturedAlignedBitmaps.clear()

        updateInterface()

        finishButton.isEnabled = false

        statusText.text =
            "Registro complementario. Retira tus accesorios si utilizas alguno y mira al frente."

        Log.d(
            TAG,
            "Registro complementario iniciado en la misma actividad."
        )
    }

    // =========================================================
    // NAVEGACIÓN AL LOGIN
    // =========================================================

    private fun goToLogin() {
        transitioningEnrollment = true

        val loginIntent = android.content.Intent(
            this,
            LoginActivity::class.java
        ).apply {
            flags =
                android.content.Intent.FLAG_ACTIVITY_CLEAR_TOP or
                        android.content.Intent.FLAG_ACTIVITY_NEW_TASK
        }

        startActivity(loginIntent)
        finish()
    }

    // =========================================================
    // PERMISOS DE CÁMARA
    // =========================================================

    private fun checkCameraPermission() {
        if (
            ContextCompat.checkSelfPermission(
                this,
                Manifest.permission.CAMERA
            ) == PackageManager.PERMISSION_GRANTED
        ) {
            startCamera()
        } else {
            cameraPermissionLauncher.launch(
                Manifest.permission.CAMERA
            )
        }
    }

    // =========================================================
    // INICIO DE CÁMARA
    // =========================================================

    private fun startCamera() {
        statusText.text = "Iniciando cámara..."

        val cameraProviderFuture =
            ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener({
            try {
                val provider = cameraProviderFuture.get()
                cameraProvider = provider

                val preview = Preview.Builder()
                    .build()
                    .also {
                        it.surfaceProvider = previewView.surfaceProvider
                    }

                val imageAnalysis = ImageAnalysis.Builder()
                    .setBackpressureStrategy(
                        ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST
                    )
                    .build()

                imageAnalysis.setAnalyzer(cameraExecutor) { imageProxy ->
                    analyzeCameraFrame(imageProxy)
                }

                provider.unbindAll()

                provider.bindToLifecycle(
                    this,
                    CameraSelector.DEFAULT_FRONT_CAMERA,
                    preview,
                    imageAnalysis
                )

                statusText.text = if (noGlassesMode) {
                    "Registro complementario. Centra tu rostro."
                } else {
                    "Registro facial. Centra tu rostro."
                }

            } catch (e: Exception) {
                statusText.text =
                    "No se pudo iniciar la cámara: ${e.message}"

                Log.e(TAG, "Error iniciando la cámara", e)
            }
        }, ContextCompat.getMainExecutor(this))
    }

    // =========================================================
    // ANALIZADOR DE FOTOGRAMAS
    // =========================================================

    private fun analyzeCameraFrame(imageProxy: ImageProxy) {

        if (
            transitioningEnrollment ||
            isFinishing ||
            isDestroyed
        ) {
            imageProxy.close()
            return
        }

        if (processingFrame) {
            imageProxy.close()
            return
        }

        processingFrame = true

        try {
            val mediaImage = imageProxy.image

            if (mediaImage == null) {
                processingFrame = false
                imageProxy.close()
                return
            }

            val image = InputImage.fromMediaImage(
                mediaImage,
                imageProxy.imageInfo.rotationDegrees
            )

            detector.process(image)
                .addOnSuccessListener { faces ->
                    if (
                        isFinishing ||
                        isDestroyed ||
                        transitioningEnrollment
                    ) {
                        return@addOnSuccessListener
                    }

                    if (faces.size == 1) {
                        val face = faces[0]
                        val yaw = face.headEulerAngleY

                        val rotation =
                            imageProxy.imageInfo.rotationDegrees

                        val frameHeight = if (
                            rotation == 90 || rotation == 270
                        ) {
                            imageProxy.width
                        } else {
                            imageProxy.height
                        }

                        val faceHeightRatio =
                            face.boundingBox.height().toFloat() /
                                    frameHeight.toFloat()

                        analyzePose(
                            yaw,
                            faceHeightRatio,
                            imageProxy,
                            face
                        )

                    } else {
                        stableFrames = 0
                        illuminationAdequate = false

                        runOnUiThread {
                            if (
                                !isFinishing &&
                                !isDestroyed &&
                                !transitioningEnrollment
                            ) {
                                statusText.text =
                                    if (faces.isEmpty()) {
                                        "No se detecta ningún rostro."
                                    } else {
                                        "Debe aparecer un solo rostro."
                                    }
                            }
                        }
                    }
                }
                .addOnFailureListener { e ->
                    Log.e(TAG, "Error analizando rostro", e)

                    stableFrames = 0
                    illuminationAdequate = false

                    runOnUiThread {
                        if (
                            !isFinishing &&
                            !isDestroyed &&
                            !transitioningEnrollment
                        ) {
                            statusText.text =
                                "Error al analizar el rostro."
                        }
                    }
                }
                .addOnCompleteListener {
                    processingFrame = false
                    imageProxy.close()
                }

        } catch (e: Exception) {
            Log.e(TAG, "Error procesando fotograma de cámara", e)

            processingFrame = false
            imageProxy.close()

            runOnUiThread {
                if (
                    !isFinishing &&
                    !isDestroyed &&
                    !transitioningEnrollment
                ) {
                    statusText.text =
                        "No se pudo procesar el fotograma."
                }
            }
        }
    }

    // =========================================================
    // EVALUACIÓN DE ILUMINACIÓN
    // =========================================================

    private fun calculateFaceBrightness(
        bitmap: Bitmap,
        faceBounds: Rect
    ): IlluminationResult? {
        if (bitmap.width <= 1 || bitmap.height <= 1) return null

        val faceWidth = faceBounds.width()
        val faceHeight = faceBounds.height()

        if (faceWidth <= 1 || faceHeight <= 1) return null

        val marginX = (faceWidth * 0.15f).toInt()
        val marginY = (faceHeight * 0.15f).toInt()

        val left = (faceBounds.left + marginX)
            .coerceIn(0, bitmap.width - 1)

        val right = (faceBounds.right - marginX)
            .coerceIn(left + 1, bitmap.width)

        val top = (faceBounds.top + marginY)
            .coerceIn(0, bitmap.height - 1)

        val bottom = (faceBounds.bottom - marginY)
            .coerceIn(top + 1, bitmap.height)

        if (right <= left || bottom <= top) return null

        val step = maxOf(
            1,
            maxOf(right - left, bottom - top) / 64
        )

        var totalBrightness = 0.0
        var pixelCount = 0
        var darkPixels = 0
        var brightPixels = 0

        for (y in top until bottom step step) {
            for (x in left until right step step) {
                val pixel = bitmap.getPixel(x, y)

                val brightness =
                    0.299 * Color.red(pixel) +
                            0.587 * Color.green(pixel) +
                            0.114 * Color.blue(pixel)

                totalBrightness += brightness
                pixelCount++

                if (brightness < DARK_PIXEL_LIMIT) darkPixels++
                if (brightness > BRIGHT_PIXEL_LIMIT) brightPixels++
            }
        }

        if (pixelCount == 0) return null

        return IlluminationResult(
            averageBrightness = totalBrightness / pixelCount,
            darkPixelRatio = darkPixels.toDouble() / pixelCount,
            brightPixelRatio = brightPixels.toDouble() / pixelCount
        )
    }

    // =========================================================
    // DETECCIÓN Y CAPTURA DE POSES
    // =========================================================

    private fun analyzePose(
        yaw: Float,
        faceHeightRatio: Float,
        imageProxy: ImageProxy,
        face: Face
    ) {
        if (
            transitioningEnrollment ||
            currentPose >= 3 ||
            isCapturingEmbedding
        ) {
            return
        }

        var frameBitmap: Bitmap? = null

        try {
            frameBitmap = imageProxyToBitmap(imageProxy)

            val lighting = calculateFaceBrightness(
                frameBitmap,
                face.boundingBox
            )

            if (lighting == null) {
                illuminationAdequate = false
                stableFrames = 0

                runOnUiThread {
                    if (
                        !isFinishing &&
                        !isDestroyed &&
                        !transitioningEnrollment
                    ) {
                        statusText.text =
                            "No se pudo evaluar la iluminación. Ajusta tu rostro."
                    }
                }
                return
            }

            Log.d(
                TAG,
                """
                ===== ILUMINACIÓN DE REGISTRO =====
                Promedio: ${String.format(Locale.US, "%.2f", lighting.averageBrightness)}
                Píxeles oscuros: ${String.format(Locale.US, "%.2f%%", lighting.darkPixelRatio * 100)}
                Píxeles sobreexpuestos: ${String.format(Locale.US, "%.2f%%", lighting.brightPixelRatio * 100)}
                Adecuada: ${lighting.isAdequate}
                Pose actual: $currentPose
                ===================================
                """.trimIndent()
            )

            if (!lighting.isAdequate) {
                illuminationAdequate = false
                stableFrames = 0

                val message = when {
                    lighting.averageBrightness < MIN_FACE_BRIGHTNESS ->
                        "Hay poca luz. Busca un lugar mejor iluminado."

                    lighting.darkPixelRatio > MAX_DARK_PIXEL_RATIO ->
                        "Hay demasiadas sombras. Ilumina mejor tu rostro."

                    lighting.brightPixelRatio > MAX_BRIGHT_PIXEL_RATIO ->
                        "Hay demasiada luz directa. Evita los reflejos."

                    else ->
                        "Ajusta la iluminación de tu rostro."
                }

                runOnUiThread {
                    if (
                        !isFinishing &&
                        !isDestroyed &&
                        !transitioningEnrollment
                    ) {
                        statusText.text = message
                    }
                }
                return
            }

            illuminationAdequate = true

            if (faceHeightRatio < 0.48f) {
                stableFrames = 0

                runOnUiThread {
                    if (
                        !isFinishing &&
                        !isDestroyed &&
                        !transitioningEnrollment
                    ) {
                        statusText.text =
                            "Acércate hasta encajar tu rostro en el óvalo."
                    }
                }
                return
            }

            if (faceHeightRatio > 0.68f) {
                stableFrames = 0

                runOnUiThread {
                    if (
                        !isFinishing &&
                        !isDestroyed &&
                        !transitioningEnrollment
                    ) {
                        statusText.text =
                            "Aléjate un poquito del óvalo."
                    }
                }
                return
            }

            val correctPose = when (currentPose) {
                0 -> abs(yaw) <= 15f
                1 -> yaw in 10f..30f
                2 -> yaw in -30f..-10f
                else -> false
            }

            if (!correctPose) {
                stableFrames = 0

                val message = when (currentPose) {
                    0 -> "Mira directamente al frente."

                    1 -> when {
                        yaw < 10f ->
                            "Gira un poquito más hacia tu izquierda."

                        yaw > 30f ->
                            "Te giraste demasiado. Vuelve un poquito."

                        else ->
                            "Ajusta ligeramente tu posición."
                    }

                    2 -> when {
                        yaw > -10f ->
                            "Gira un poquito más hacia tu derecha."

                        yaw < -30f ->
                            "Te giraste demasiado. Vuelve un poquito."

                        else ->
                            "Ajusta ligeramente tu posición."
                    }

                    else ->
                        "Registro completado."
                }

                runOnUiThread {
                    if (
                        !isFinishing &&
                        !isDestroyed &&
                        !transitioningEnrollment
                    ) {
                        statusText.text = message
                    }
                }
                return
            }

            stableFrames++

            runOnUiThread {
                if (
                    !isFinishing &&
                    !isDestroyed &&
                    !transitioningEnrollment
                ) {
                    statusText.text =
                        "Iluminación adecuada. Mantén la posición... ($stableFrames/$REQUIRED_STABLE_FRAMES)"
                }
            }

            if (stableFrames < REQUIRED_STABLE_FRAMES) return

            isCapturingEmbedding = true
            stableFrames = 0

            val poseToCapture = currentPose

            runOnUiThread {
                if (
                    !isFinishing &&
                    !isDestroyed &&
                    !transitioningEnrollment
                ) {
                    statusText.text =
                        "Capturando y procesando rostro..."
                }
            }

            val bitmapForProcessing = frameBitmap
                ?: throw IllegalStateException(
                    "No se pudo preparar la imagen."
                )

            frameBitmap = null

            Log.d(
                TAG,
                "Bitmap: ${bitmapForProcessing.width}x${bitmapForProcessing.height}"
            )

            Log.d(
                TAG,
                "Rotación: ${imageProxy.imageInfo.rotationDegrees}"
            )

            Log.d(
                TAG,
                "Ojo izquierdo ML Kit: ${
                    face.getLandmark(FaceLandmark.LEFT_EYE)?.position
                }"
            )

            Log.d(
                TAG,
                "Ojo derecho ML Kit: ${
                    face.getLandmark(FaceLandmark.RIGHT_EYE)?.position
                }"
            )

            Log.d(
                TAG,
                "Nariz ML Kit: ${
                    face.getLandmark(FaceLandmark.NOSE_BASE)?.position
                }"
            )

            try {
                embeddingExecutor.execute {
                    processCapturedPose(
                        bitmapForProcessing,
                        face,
                        poseToCapture
                    )
                }
            } catch (e: Exception) {
                bitmapForProcessing.recycle()
                isCapturingEmbedding = false
                throw e
            }

        } catch (e: Exception) {
            Log.e(TAG, "Error capturando fotograma", e)

            runOnUiThread {
                if (
                    !isFinishing &&
                    !isDestroyed &&
                    !transitioningEnrollment
                ) {
                    statusText.text =
                        "No se pudo capturar el rostro. Inténtalo de nuevo."
                }
            }

            isCapturingEmbedding = false

        } finally {
            frameBitmap?.let {
                if (!it.isRecycled) it.recycle()
            }
        }
    }

    // =========================================================
    // PROCESAMIENTO DE EMBEDDINGS
    // =========================================================

    private fun processCapturedPose(
        bitmap: Bitmap,
        face: Face,
        poseToCapture: Int
    ) {
        var mediaPipeAligned: Bitmap? = null
        var mlKitAligned: Bitmap? = null
        var diagnosticBitmap: Bitmap? = null

        try {
            val processor = embeddingProcessor
                ?: throw IllegalStateException(
                    "Motor biométrico no disponible."
                )

            val detection = mediaPipeDetector?.detect(bitmap)

            Log.d(TAG, "===== RESULTADO MEDIAPIPE =====")

            if (detection != null) {
                Log.d(TAG, "Confianza: ${detection.confidence}")
                Log.d(TAG, "Caja: ${detection.boundingBox}")
                Log.d(TAG, "Puntos: ${detection.keypoints}")
            }

            if (
                detection == null ||
                detection.confidence < 0.5f ||
                detection.keypoints.size < 3
            ) {
                throw IllegalStateException(
                    "MediaPipe no detectó correctamente el rostro."
                )
            }

            mediaPipeAligned =
                alignEnrollmentFace(bitmap, detection)
                    ?: throw IllegalStateException(
                        "Falló la alineación MediaPipe."
                    )

            mlKitAligned =
                alignEnrollmentFace(bitmap, face)
                    ?: throw IllegalStateException(
                        "Falló la alineación ML Kit."
                    )

            val mediaPipeEmbedding =
                processor.generateEmbedding(mediaPipeAligned)

            val mlKitEmbedding =
                processor.generateEmbedding(mlKitAligned)

            if (
                mediaPipeEmbedding.size != 128 ||
                mlKitEmbedding.size != 128 ||
                mediaPipeEmbedding.any { !it.isFinite() } ||
                mlKitEmbedding.any { !it.isFinite() }
            ) {
                throw IllegalStateException(
                    "Uno de los embeddings no es válido."
                )
            }

            val alignmentSimilarity = cosineSimilarity(
                mediaPipeEmbedding,
                mlKitEmbedding
            )

            val mediaPipeNorm = sqrt(
                mediaPipeEmbedding.sumOf {
                    (it * it).toDouble()
                }
            )

            val mlKitNorm = sqrt(
                mlKitEmbedding.sumOf {
                    (it * it).toDouble()
                }
            )

            Log.d(TAG, "===== COMPARACIÓN A/B =====")
            Log.d(
                TAG,
                "Similitud MediaPipe vs ML Kit: $alignmentSimilarity"
            )
            Log.d(TAG, "Norma MediaPipe: $mediaPipeNorm")
            Log.d(TAG, "Norma ML Kit: $mlKitNorm")

            diagnosticBitmap = mediaPipeAligned.copy(
                Bitmap.Config.ARGB_8888,
                false
            )

            runOnUiThread {
                if (
                    !isFinishing &&
                    !isDestroyed &&
                    !transitioningEnrollment &&
                    currentPose == poseToCapture
                ) {
                    capturedEmbeddings.add(mediaPipeEmbedding)
                    capturedMlKitEmbeddings.add(mlKitEmbedding)

                    capturedAlignedBitmaps.add(
                        diagnosticBitmap!!
                    )
                    diagnosticBitmap = null

                    currentPose++
                    stableFrames = 0

                    updateInterface()

                    statusText.text =
                        "Pose ${poseToCapture + 1} registrada correctamente."

                    Log.d(
                        TAG,
                        "Pose ${poseToCapture + 1} capturada."
                    )

                    Log.d(
                        TAG,
                        "Total MediaPipe: ${capturedEmbeddings.size}"
                    )

                    Log.d(
                        TAG,
                        "Total ML Kit: ${capturedMlKitEmbeddings.size}"
                    )

                    if (
                        capturedEmbeddings.size == 3 &&
                        capturedMlKitEmbeddings.size == 3
                    ) {
                        logEmbeddingDiagnostics()
                        logAlignmentComparisonDiagnostics()
                        runTwoImageDiagnostics()
                    }

                } else {
                    diagnosticBitmap?.recycle()
                    diagnosticBitmap = null
                }

                isCapturingEmbedding = false
            }

        } catch (e: Exception) {
            Log.e(TAG, "Error procesando embedding", e)

            runOnUiThread {
                if (
                    !isFinishing &&
                    !isDestroyed &&
                    !transitioningEnrollment
                ) {
                    statusText.text =
                        "No se pudo capturar el rostro. Inténtalo de nuevo."
                }

                isCapturingEmbedding = false
            }

        } finally {
            mediaPipeAligned?.let {
                if (!it.isRecycled) it.recycle()
            }

            mlKitAligned?.let {
                if (!it.isRecycled) it.recycle()
            }

            diagnosticBitmap?.let {
                if (!it.isRecycled) it.recycle()
            }

            if (!bitmap.isRecycled) bitmap.recycle()
        }
    }

    // =========================================================
    // DIAGNÓSTICO DE EMBEDDINGS
    // =========================================================

    private fun logEmbeddingDiagnostics() {
        if (capturedEmbeddings.size != 3) return

        val front = capturedEmbeddings[0]
        val left = capturedEmbeddings[1]
        val right = capturedEmbeddings[2]

        Log.d(
            TAG,
            "===== PRUEBA DE EMBEDDINGS MEDIAPIPE ====="
        )

        capturedEmbeddings.forEachIndexed { index, embedding ->
            val norm = sqrt(
                embedding.sumOf {
                    (it * it).toDouble()
                }
            )

            Log.d(
                TAG,
                "Pose ${index + 1}: dimensiones=${embedding.size}, norma=$norm"
            )
        }

        Log.d(
            TAG,
            "Similitud frontal-izquierda: ${cosineSimilarity(front, left)}"
        )

        Log.d(
            TAG,
            "Similitud frontal-derecha: ${cosineSimilarity(front, right)}"
        )

        Log.d(
            TAG,
            "Similitud izquierda-derecha: ${cosineSimilarity(left, right)}"
        )
    }

    private fun logAlignmentComparisonDiagnostics() {
        Log.d(
            TAG,
            "========== COMPARACIÓN ENTRE MÉTODOS =========="
        )

        if (
            capturedEmbeddings.size != 3 ||
            capturedMlKitEmbeddings.size != 3
        ) {
            Log.e(TAG, "No hay tres embeddings de cada método.")
            return
        }

        fun logComparisons(
            method: String,
            embeddings: List<FloatArray>
        ) {
            Log.d(TAG, "----- $method -----")

            Log.d(
                TAG,
                "Frontal - izquierda: ${cosineSimilarity(embeddings[0], embeddings[1])}"
            )

            Log.d(
                TAG,
                "Frontal - derecha: ${cosineSimilarity(embeddings[0], embeddings[2])}"
            )

            Log.d(
                TAG,
                "Izquierda - derecha: ${cosineSimilarity(embeddings[1], embeddings[2])}"
            )
        }

        logComparisons("MEDIAPIPE", capturedEmbeddings)
        logComparisons("ML KIT", capturedMlKitEmbeddings)
    }

    private fun runTwoImageDiagnostics() {
        if (capturedAlignedBitmaps.size != 3) return

        val images = try {
            capturedAlignedBitmaps.map {
                it.copy(Bitmap.Config.ARGB_8888, false)
            }
        } catch (e: Exception) {
            Log.e(
                TAG,
                "No se pudieron copiar las imágenes de diagnóstico",
                e
            )
            return
        }

        try {
            embeddingExecutor.execute {
                try {
                    val processor = embeddingProcessor
                        ?: throw IllegalStateException(
                            "Motor biométrico no disponible."
                        )

                    val pairs = listOf(
                        Triple(0, 1, "FRONTAL - IZQUIERDA"),
                        Triple(0, 2, "FRONTAL - DERECHA"),
                        Triple(1, 2, "IZQUIERDA - DERECHA")
                    )

                    for ((first, second, label) in pairs) {
                        val result = processor.generateEmbeddings(
                            images[first],
                            images[second]
                        )

                        val similarity = cosineSimilarity(
                            result.first,
                            result.second
                        )

                        Log.d(TAG, "===== $label =====")
                        Log.d(
                            TAG,
                            "Vector 1: ${result.first.take(10)}"
                        )
                        Log.d(
                            TAG,
                            "Vector 2: ${result.second.take(10)}"
                        )
                        Log.d(TAG, "Similitud: $similarity")
                    }

                    runOnUiThread {
                        if (
                            !isFinishing &&
                            !isDestroyed &&
                            !transitioningEnrollment
                        ) {
                            statusText.text =
                                "Pruebas de dos imágenes completadas."
                        }
                    }

                } catch (e: Exception) {
                    Log.e(
                        TAG,
                        "Error en pruebas de dos imágenes",
                        e
                    )
                } finally {
                    images.forEach {
                        if (!it.isRecycled) it.recycle()
                    }
                }
            }
        } catch (e: Exception) {
            images.forEach {
                if (!it.isRecycled) it.recycle()
            }

            Log.e(
                TAG,
                "No se pudo iniciar el diagnóstico",
                e
            )
        }
    }

    // =========================================================
    // ACTUALIZACIÓN DE INTERFAZ
    // =========================================================

    private fun updateInterface() {
        progress.progress = currentPose

        progressText.text =
            "$currentPose de 3 poses procesadas"

        val registrationLabel = if (noGlassesMode) {
            "REGISTRO COMPLEMENTARIO · "
        } else {
            "REGISTRO PRINCIPAL · "
        }

        poseText.text = when (currentPose) {
            0 -> registrationLabel + "01 · MIRA AL FRENTE"
            1 -> registrationLabel + "02 · GIRA A LA IZQUIERDA"
            2 -> registrationLabel + "03 · GIRA A LA DERECHA"
            else -> if (noGlassesMode) {
                "REGISTRO COMPLEMENTARIO COMPLETADO"
            } else {
                "REGISTRO PRINCIPAL COMPLETADO"
            }
        }

        finishButton.isEnabled =
            currentPose == 3 &&
                    capturedEmbeddings.size == 3 &&
                    capturedMlKitEmbeddings.size == 3

        if (currentPose == 3) {
            statusText.text = if (noGlassesMode) {
                "¡Registro complementario completado!"
            } else {
                "¡Registro principal completado!"
            }
        }
    }

    // =========================================================
    // CONVERSIÓN DE IMAGEN DE CÁMARA
    // =========================================================

    private fun imageProxyToBitmap(
        imageProxy: ImageProxy
    ): Bitmap {
        val width = imageProxy.width and -2
        val height = imageProxy.height and -2

        require(width > 0 && height > 0) {
            "Dimensiones de cámara no válidas."
        }

        val planes = imageProxy.planes

        require(planes.size >= 3) {
            "La imagen no contiene los tres planos YUV."
        }

        val yPlane = planes[0]
        val uPlane = planes[1]
        val vPlane = planes[2]

        val yBuffer = yPlane.buffer.duplicate()
        val uBuffer = uPlane.buffer.duplicate()
        val vBuffer = vPlane.buffer.duplicate()

        val yBase = yBuffer.position()
        val uBase = uBuffer.position()
        val vBase = vBuffer.position()

        val chromaWidth = width / 2
        val chromaHeight = height / 2

        val nv21 = ByteArray(
            width * height + width * height / 2
        )

        var outputIndex = 0

        for (row in 0 until height) {
            for (col in 0 until width) {
                val index =
                    yBase +
                            row * yPlane.rowStride +
                            col * yPlane.pixelStride

                nv21[outputIndex++] = yBuffer.get(index)
            }
        }

        for (row in 0 until chromaHeight) {
            for (col in 0 until chromaWidth) {
                val uIndex =
                    uBase +
                            row * uPlane.rowStride +
                            col * uPlane.pixelStride

                val vIndex =
                    vBase +
                            row * vPlane.rowStride +
                            col * vPlane.pixelStride

                nv21[outputIndex++] = vBuffer.get(vIndex)
                nv21[outputIndex++] = uBuffer.get(uIndex)
            }
        }

        val yuvImage = YuvImage(
            nv21,
            ImageFormat.NV21,
            width,
            height,
            null
        )

        val outputStream = ByteArrayOutputStream()

        val compressed = yuvImage.compressToJpeg(
            Rect(0, 0, width, height),
            95,
            outputStream
        )

        check(compressed) {
            "No se pudo comprimir la imagen YUV."
        }

        val bytes = outputStream.toByteArray()

        val original = BitmapFactory.decodeByteArray(
            bytes,
            0,
            bytes.size
        ) ?: throw IllegalStateException(
            "No se pudo convertir ImageProxy a Bitmap."
        )

        val rotation = imageProxy.imageInfo.rotationDegrees

        if (rotation == 0) return original

        val matrix = Matrix().apply {
            postRotate(rotation.toFloat())
        }

        val rotated = Bitmap.createBitmap(
            original,
            0,
            0,
            original.width,
            original.height,
            matrix,
            true
        )

        original.recycle()
        return rotated
    }

    // =========================================================
    // ALINEACIÓN FACIAL ML KIT
    // =========================================================

    private fun alignEnrollmentFace(
        bitmap: Bitmap,
        face: Face
    ): Bitmap? {
        val leftEye = face.getLandmark(
            FaceLandmark.LEFT_EYE
        ) ?: return null

        val rightEye = face.getLandmark(
            FaceLandmark.RIGHT_EYE
        ) ?: return null

        val nose = face.getLandmark(
            FaceLandmark.NOSE_BASE
        ) ?: return null

        val sourcePoints = floatArrayOf(
            leftEye.position.x,
            leftEye.position.y,
            rightEye.position.x,
            rightEye.position.y,
            nose.position.x,
            nose.position.y
        )

        Log.d(
            TAG,
            "Coordenadas ML Kit: ${sourcePoints.contentToString()}"
        )

        val targetPoints = floatArrayOf(
            38.3f, 51.7f,
            73.5f, 51.5f,
            56.0f, 71.7f
        )

        val matrix = Matrix()

        if (!matrix.setPolyToPoly(
                sourcePoints,
                0,
                targetPoints,
                0,
                3
            )
        ) {
            return null
        }

        val alignedBitmap = Bitmap.createBitmap(
            112,
            112,
            Bitmap.Config.ARGB_8888
        )

        Canvas(alignedBitmap).drawBitmap(
            bitmap,
            matrix,
            Paint(
                Paint.ANTI_ALIAS_FLAG or
                        Paint.FILTER_BITMAP_FLAG
            )
        )

        return alignedBitmap
    }

    // =========================================================
    // ALINEACIÓN FACIAL MEDIAPIPE
    // =========================================================

    private fun alignEnrollmentFace(
        bitmap: Bitmap,
        detection: MediaPipeDetection
    ): Bitmap? {
        if (detection.keypoints.size < 3) {
            Log.e(TAG, "Faltan puntos faciales de MediaPipe.")
            return null
        }

        val eyes = detection.keypoints
            .take(2)
            .sortedBy { it.x }

        val leftEye = eyes[0]
        val rightEye = eyes[1]
        val nose = detection.keypoints[2]

        val sourcePoints = floatArrayOf(
            leftEye.x,
            leftEye.y,
            rightEye.x,
            rightEye.y,
            nose.x,
            nose.y
        )

        val targetPoints = floatArrayOf(
            38.3f, 51.7f,
            73.5f, 51.5f,
            56.0f, 71.7f
        )

        val matrix = Matrix()

        if (!matrix.setPolyToPoly(
                sourcePoints,
                0,
                targetPoints,
                0,
                3
            )
        ) {
            return null
        }

        val alignedBitmap = Bitmap.createBitmap(
            112,
            112,
            Bitmap.Config.ARGB_8888
        )

        Canvas(alignedBitmap).drawBitmap(
            bitmap,
            matrix,
            Paint(
                Paint.ANTI_ALIAS_FLAG or
                        Paint.FILTER_BITMAP_FLAG
            )
        )

        Log.d(TAG, "Alineación MediaPipe completada.")

        return alignedBitmap
    }

    // =========================================================
    // SIMILITUD COSENO
    // =========================================================

    private fun cosineSimilarity(
        a: FloatArray,
        b: FloatArray
    ): Double {
        require(a.size == b.size) {
            "Los embeddings tienen dimensiones diferentes."
        }

        var dotProduct = 0.0
        var normA = 0.0
        var normB = 0.0

        for (i in a.indices) {
            require(a[i].isFinite() && b[i].isFinite()) {
                "Embedding contiene valores inválidos."
            }

            dotProduct +=
                a[i].toDouble() * b[i].toDouble()

            normA +=
                a[i].toDouble() * a[i].toDouble()

            normB +=
                b[i].toDouble() * b[i].toDouble()
        }

        if (normA <= 0.0 || normB <= 0.0) {
            throw IllegalArgumentException(
                "No se puede calcular la similitud de embeddings nulos."
            )
        }

        return (
                dotProduct /
                        (sqrt(normA) * sqrt(normB))
                ).coerceIn(-1.0, 1.0)
    }

    // =========================================================
    // LIBERACIÓN DE RECURSOS
    // =========================================================

    override fun onDestroy() {
        transitioningEnrollment = true

        cameraProvider?.unbindAll()
        cameraProvider = null

        cameraExecutor.shutdownNow()

        try {
            detector.close()
        } catch (e: Exception) {
            Log.e(TAG, "Error cerrando ML Kit", e)
        }

        if (!embeddingExecutor.isShutdown) {
            try {
                embeddingExecutor.execute {
                    try {
                        embeddingProcessor?.close()
                        embeddingProcessor = null

                        mediaPipeDetector?.close()
                        mediaPipeDetector = null

                        capturedAlignedBitmaps.forEach {
                            if (!it.isRecycled) {
                                it.recycle()
                            }
                        }

                        capturedAlignedBitmaps.clear()
                        capturedEmbeddings.clear()
                        capturedMlKitEmbeddings.clear()

                    } catch (e: Exception) {
                        Log.e(
                            TAG,
                            "Error liberando motores",
                            e
                        )
                    }
                }

                embeddingExecutor.shutdown()

            } catch (e: Exception) {
                Log.e(TAG, "Error cerrando executor", e)
                embeddingExecutor.shutdownNow()
            }
        }

        super.onDestroy()
    }
}