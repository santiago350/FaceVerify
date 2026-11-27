package com.example.myapplication

import android.Manifest
import android.content.Intent
import android.content.SharedPreferences
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.ImageFormat
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.PointF
import android.graphics.Rect
import android.graphics.YuvImage
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.util.Log
import android.view.View
import android.widget.Button
import android.widget.LinearLayout
import android.widget.ProgressBar
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import com.google.android.gms.tasks.Tasks
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetector
import com.google.mlkit.vision.face.FaceDetectorOptions
import java.io.ByteArrayOutputStream
import java.util.Locale
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import kotlin.math.abs
import kotlin.random.Random
import androidx.appcompat.app.AlertDialog
class MainActivity : AppCompatActivity() {

    companion object {
        private const val TAG = "FACEVERIFY"

        private const val THRESHOLD = 0.6

        private const val CENTER_THRESHOLD = 0.18
        private const val TURN_THRESHOLD = 0.14
        private const val REQUIRED_FRAMES = 3

        private const val UP_PITCH_THRESHOLD = -10f
        private const val DOWN_PITCH_THRESHOLD = 10f
        private const val CENTER_PITCH_THRESHOLD = 15f

        private const val MAX_FACE_LOST_FRAMES = 12

        private const val MIN_FACE_BRIGHTNESS = 95.0
        private const val MAX_DARK_PIXEL_RATIO = 0.20
        private const val MAX_BRIGHT_PIXEL_RATIO = 0.20

        private const val DARK_PIXEL_LIMIT = 55.0
        private const val BRIGHT_PIXEL_LIMIT = 245.0

        private const val MAX_FACE_FAILED_ATTEMPTS = 3

        private const val RETRY_COOLDOWN_SECONDS = 15L
        private const val RETRY_PREFERENCES = "FaceVerifyRetryCooldown"

        // NUEVOS TIEMPOS
        private const val VERIFICATION_TIMEOUT_SECONDS = 30L
        private const val STEP_TIMEOUT_SECONDS = 10L
        private const val TIMER_CHECK_INTERVAL_MS = 250L
    }

    private lateinit var previewView: PreviewView
    private lateinit var txtCameraStatus: TextView
    private lateinit var txtChallenge: TextView
    private lateinit var txtResult: TextView
    private lateinit var txtProgress: TextView
    private lateinit var progressLiveness: ProgressBar
    private lateinit var btnCompare: Button

    private lateinit var resultButtons: LinearLayout
    private lateinit var btnBackMenu: Button
    private lateinit var btnRestartVerification: Button

    private val cameraExecutor: ExecutorService =
        Executors.newSingleThreadExecutor()

    private val faceExecutor: ExecutorService =
        Executors.newSingleThreadExecutor()

    private val retryHandler = Handler(Looper.getMainLooper())
    private val timeoutHandler = Handler(Looper.getMainLooper())

    private lateinit var retryPreferences: SharedPreferences

    private var retryUntilMillis = 0L

    // Temporizadores de verificación
    @Volatile
    private var verificationTimedOut = false

    @Volatile
    private var verificationDeadline = 0L

    @Volatile
    private var stepDeadline = 0L

    private var timeoutRunnable: Runnable? = null

    private val retryCountdownRunnable = object : Runnable {
        override fun run() {
            if (!::btnRestartVerification.isInitialized) return

            val remaining =
                retryUntilMillis - System.currentTimeMillis()

            if (remaining > 0L) {
                val seconds = (remaining + 999L) / 1000L

                btnRestartVerification.text =
                    "REINTENTAR EN ${seconds}s"

                btnRestartVerification.isEnabled = false

                retryHandler.postDelayed(this, 250L)
            } else {
                btnRestartVerification.text =
                    "INICIAR VERIFICACIÓN"

                btnRestartVerification.isEnabled = true
            }
        }
    }

    private var cameraProvider: ProcessCameraProvider? = null

    private var mediaPipeDetector: MediaPipeFaceDetector? = null
    private var embeddingProcessor: FaceEmbeddingProcessor? = null
    private var blinkFaceDetector: FaceDetector? = null

    private val blinkLivenessDetector = BlinkLivenessDetector()

    private var username = ""

    @Volatile
    private var isChallengeActive = false

    @Volatile
    private var isVerificationRunning = false

    @Volatile
    private var verificationFinished = false

    @Volatile
    private var illuminationAdequate = false

    private var challengeStep = 0
    private var stableFrames = 0
    private var faceLostFrames = 0
    private var firstTurnDirection = 1

    private enum class RandomAction {
        BLINK, VERTICAL
    }

    private enum class VerticalDirection {
        UP, DOWN
    }

    private data class FaceSignals(
        val blinkStatus: BlinkLivenessDetector.Status?,
        val pitch: Float
    )

    private var randomizedActions = emptyList<RandomAction>()
    private var verticalDirection = VerticalDirection.UP

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

    private val requestCameraPermission =
        registerForActivityResult(
            ActivityResultContracts.RequestPermission()
        ) { granted ->
            if (granted) {
                startCamera()
            } else {
                txtCameraStatus.text =
                    "Se necesita permiso para utilizar la cámara."
            }
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        initializeViews()

        username = intent.getStringExtra("username") ?: ""

        retryPreferences = getSharedPreferences(
            RETRY_PREFERENCES,
            MODE_PRIVATE
        )

        if (username.isBlank()) {
            txtResult.text =
                "No se recibió el usuario de inicio de sesión."

            btnCompare.isEnabled = false
            return
        }

        initializeBiometricEngines()

        btnCompare.setOnClickListener {
            startLiveness()
        }

        checkCameraPermission()
    }

    // ---------------------------------------------------------
    // INTERFAZ
    // ---------------------------------------------------------

    private fun initializeViews() {
        previewView = findViewById(R.id.previewView)
        txtCameraStatus = findViewById(R.id.txtCameraStatus)
        txtChallenge = findViewById(R.id.txtChallenge)
        txtResult = findViewById(R.id.txtResult)
        txtProgress = findViewById(R.id.txtProgress)
        progressLiveness = findViewById(R.id.progressLiveness)
        btnCompare = findViewById(R.id.btnCompare)

        resultButtons = findViewById(R.id.resultButtons)
        btnBackMenu = findViewById(R.id.btnBackMenu)
        btnRestartVerification =
            findViewById(R.id.btnRestartVerification)

        resultButtons.visibility = View.GONE

        btnBackMenu.setOnClickListener {
            returnToUserMenu()
        }

        btnRestartVerification.setOnClickListener {
            restartVerification()
        }

        resetLivenessProgress()
    }

    private fun returnToUserMenu() {
        cancelVerificationTimers()

        retryHandler.removeCallbacks(
            retryCountdownRunnable
        )

        isChallengeActive = false
        isVerificationRunning = false
        verificationFinished = true

        cameraProvider?.unbindAll()

        val intent = Intent(
            this,
            UserMenuActivity::class.java
        ).apply {
            putExtra("username", username)

            flags =
                Intent.FLAG_ACTIVITY_CLEAR_TOP or
                        Intent.FLAG_ACTIVITY_SINGLE_TOP
        }

        startActivity(intent)
        finish()
    }

    private fun showResultButtons(
        accepted: Boolean,
        technicalError: Boolean = false,
        lockoutStatus: UserDatabase.LockoutStatus? = null
    ) {
        runOnUiThread {
            btnCompare.visibility = View.GONE
            resultButtons.visibility = View.VISIBLE

            btnBackMenu.isEnabled = true
            btnBackMenu.text = "VOLVER AL MENÚ ANTERIOR"

            if (accepted) {
                clearRetryCooldown()

                btnRestartVerification.text =
                    "INICIAR VERIFICACIÓN"

                btnRestartVerification.isEnabled = true

            } else if (technicalError) {
                clearRetryCooldown()

                btnRestartVerification.text =
                    "REINTENTAR VERIFICACIÓN"

                btnRestartVerification.isEnabled = true

            } else if (
                lockoutStatus != null &&
                lockoutStatus.isLocked
            ) {
                setRetryCooldownMillis(
                    lockoutStatus.remainingMillis
                )

            } else {
                clearRetryCooldown()

                btnRestartVerification.text =
                    "INICIAR VERIFICACIÓN"

                btnRestartVerification.isEnabled = true
            }
        }
    }

    // ---------------------------------------------------------
    // TEMPORIZADORES: 30 SEGUNDOS TOTAL, 10 POR PASO
    // ---------------------------------------------------------

    private fun startVerificationTimers() {
        cancelVerificationTimers()

        verificationTimedOut = false

        verificationDeadline =
            System.currentTimeMillis() +
                    VERIFICATION_TIMEOUT_SECONDS * 1000L

        resetStepTimer()

        scheduleTimeoutCheck()
    }

    private fun resetStepTimer() {
        stepDeadline =
            System.currentTimeMillis() +
                    STEP_TIMEOUT_SECONDS * 1000L
    }

    private fun scheduleTimeoutCheck() {
        timeoutRunnable?.let {
            timeoutHandler.removeCallbacks(it)
        }

        val runnable = object : Runnable {
            override fun run() {
                if (
                    verificationFinished ||
                    verificationTimedOut ||
                    (!isChallengeActive && !isVerificationRunning)
                ) {
                    return
                }

                val now = System.currentTimeMillis()

                if (now >= verificationDeadline) {
                    timeoutVerification(
                        "Se agotaron los 30 segundos disponibles para la verificación."
                    )
                    return
                }

                if (
                    isChallengeActive &&
                    now >= stepDeadline
                ) {
                    timeoutVerification(
                        "No se completó el paso dentro de los 10 segundos permitidos."
                    )
                    return
                }

                timeoutHandler.postDelayed(
                    this,
                    TIMER_CHECK_INTERVAL_MS
                )
            }
        }

        timeoutRunnable = runnable

        timeoutHandler.postDelayed(
            runnable,
            TIMER_CHECK_INTERVAL_MS
        )
    }

    private fun cancelVerificationTimers() {
        timeoutRunnable?.let {
            timeoutHandler.removeCallbacks(it)
        }

        timeoutRunnable = null
        timeoutHandler.removeCallbacksAndMessages(null)

        verificationDeadline = 0L
        stepDeadline = 0L
    }

    private fun timeoutVerification(message: String) {
        if (verificationTimedOut) return

        verificationTimedOut = true
        verificationFinished = true
        isChallengeActive = false
        isVerificationRunning = false

        stableFrames = 0
        challengeStep = 0
        faceLostFrames = 0

        blinkLivenessDetector.reset()

        cancelVerificationTimers()

        Log.w(
            TAG,
            "VERIFICACIÓN CANCELADA POR TIEMPO: $message"
        )

        runOnUiThread {
            if (isFinishing || isDestroyed) return@runOnUiThread

            // Detener completamente la cámara
            cameraProvider?.unbindAll()

            // Actualizar interfaz
            btnCompare.isEnabled = false
            btnCompare.visibility = View.GONE

            resultButtons.visibility = View.GONE

            txtResult.text =
                "Verificación no completada."

            txtChallenge.text =
                "No se completó la verificación."

            txtCameraStatus.text =
                "Proceso finalizado."

            txtProgress.text =
                "VERIFICACIÓN NO COMPLETADA"

            progressLiveness.progress = 0

            // Mostrar mensaje y esperar que el usuario pulse OK
            AlertDialog.Builder(this)
                .setTitle("Verificación no completada")
                .setMessage(
                    "No se detectó correctamente el rostro " +
                            "o no se completaron las comprobaciones " +
                            "dentro del tiempo disponible.\n\n" +
                            "Por seguridad, volverás al menú principal."
                )
                .setCancelable(false)
                .setPositiveButton("OK") { _, _ ->
                    returnToUserMenu()
                }
                .show()
        }
    }

    // ---------------------------------------------------------
    // BLOQUEO DE REINTENTOS
    // ---------------------------------------------------------

    private fun setRetryCooldownMillis(milliseconds: Long) {
        retryUntilMillis =
            System.currentTimeMillis() +
                    milliseconds.coerceAtLeast(0L)

        retryPreferences.edit()
            .putLong(
                retryKey(),
                retryUntilMillis
            )
            .apply()

        updateRetryCountdown()
    }

    private fun restartVerification() {
        if (isChallengeActive || isVerificationRunning) {
            return
        }

        val remaining = getRetryRemaining()

        if (remaining > 0L) {
            updateRetryCountdown()
            return
        }

        val lockoutStatus = getFaceLockoutStatus()

        if (lockoutStatus == null) {
            txtResult.text =
                "No se pudo consultar el bloqueo de seguridad."
            return
        }

        if (lockoutStatus.isLocked) {
            txtResult.text =
                "BLOQUEO DE SEGURIDAD ACTIVO\n\n" +
                        "Tiempo restante: " +
                        formatRemainingTime(
                            lockoutStatus.remainingMillis
                        )

            btnRestartVerification.isEnabled = false

            retryUntilMillis =
                System.currentTimeMillis() +
                        lockoutStatus.remainingMillis

            retryHandler.removeCallbacks(
                retryCountdownRunnable
            )

            retryHandler.post(
                retryCountdownRunnable
            )

            return
        }

        resultButtons.visibility = View.GONE
        btnCompare.visibility = View.VISIBLE

        verificationFinished = false
        verificationTimedOut = false
        isVerificationRunning = false
        isChallengeActive = false

        challengeStep = 0
        stableFrames = 0
        faceLostFrames = 0

        blinkLivenessDetector.reset()

        resetLivenessProgress()

        txtResult.text =
            "Preparando nueva verificación..."

        txtChallenge.text =
            "Coloca tu rostro frente a la cámara."

        btnCompare.isEnabled = illuminationAdequate

        if (illuminationAdequate) {
            startLiveness()
        } else {
            txtCameraStatus.text =
                "Esperando iluminación adecuada."

            btnCompare.isEnabled = false
        }
    }

    private fun retryKey(): String =
        "retry_until_$username"

    private fun getRetryRemaining(): Long {
        val until = retryPreferences.getLong(
            retryKey(),
            0L
        )

        return (until - System.currentTimeMillis())
            .coerceAtLeast(0L)
    }

    private fun setRetryCooldown(seconds: Long) {
        retryUntilMillis =
            System.currentTimeMillis() +
                    seconds * 1000L

        retryPreferences.edit()
            .putLong(
                retryKey(),
                retryUntilMillis
            )
            .apply()

        updateRetryCountdown()
    }

    private fun clearRetryCooldown() {
        retryUntilMillis = 0L

        retryPreferences.edit()
            .remove(retryKey())
            .apply()

        retryHandler.removeCallbacks(
            retryCountdownRunnable
        )
    }

    private fun updateRetryCountdown() {
        retryHandler.removeCallbacks(
            retryCountdownRunnable
        )

        retryUntilMillis =
            retryPreferences.getLong(
                retryKey(),
                0L
            )

        retryCountdownRunnable.run()
    }

    private fun formatRemainingTime(
        milliseconds: Long
    ): String {
        val totalSeconds =
            ((milliseconds + 999L) / 1000L)
                .coerceAtLeast(1L)

        return if (totalSeconds >= 60L) {
            val minutes = totalSeconds / 60L
            val seconds = totalSeconds % 60L

            "${minutes} min ${seconds} s"
        } else {
            "$totalSeconds segundos"
        }
    }

    // ---------------------------------------------------------
    // PROGRESO
    // ---------------------------------------------------------

    private fun resetLivenessProgress() {
        runOnUiThread {
            txtProgress.text = "PASO 1 DE 5"
            progressLiveness.progress = 0
        }
    }

    private fun updateLivenessProgress(step: Int) {
        val safeStep = step.coerceIn(1, 5)

        runOnUiThread {
            txtProgress.text =
                "PASO $safeStep DE 5"

            progressLiveness.progress = safeStep
        }
    }

    // ---------------------------------------------------------
    // MOTORES BIOMÉTRICOS
    // ---------------------------------------------------------

    private fun initializeBiometricEngines() {
        try {
            mediaPipeDetector =
                MediaPipeFaceDetector(this)

            embeddingProcessor =
                FaceEmbeddingProcessor(this)

            val blinkOptions =
                FaceDetectorOptions.Builder()
                    .setPerformanceMode(
                        FaceDetectorOptions.PERFORMANCE_MODE_FAST
                    )
                    .setClassificationMode(
                        FaceDetectorOptions.CLASSIFICATION_MODE_ALL
                    )
                    .setMinFaceSize(0.10f)
                    .build()

            blinkFaceDetector =
                FaceDetection.getClient(blinkOptions)

            Log.d(
                TAG,
                "Motores biométricos inicializados"
            )

            txtResult.text =
                "Usuario: $username\n\nMotores biométricos listos."

        } catch (e: Exception) {
            Log.e(
                TAG,
                "Error inicializando motores",
                e
            )

            txtResult.text =
                "Error inicializando motores: ${e.localizedMessage}"

            btnCompare.isEnabled = false
        }
    }

    // ---------------------------------------------------------
    // PERMISOS Y CÁMARA
    // ---------------------------------------------------------

    private fun checkCameraPermission() {
        val granted =
            ContextCompat.checkSelfPermission(
                this,
                Manifest.permission.CAMERA
            ) == PackageManager.PERMISSION_GRANTED

        if (granted) {
            startCamera()
        } else {
            requestCameraPermission.launch(
                Manifest.permission.CAMERA
            )
        }
    }

    private fun startCamera() {
        val cameraProviderFuture =
            ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener({
            try {
                cameraProvider =
                    cameraProviderFuture.get()

                val preview = Preview.Builder().build()

                preview.setSurfaceProvider(
                    previewView.surfaceProvider
                )

                val imageAnalysis =
                    ImageAnalysis.Builder()
                        .setBackpressureStrategy(
                            ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST
                        )
                        .build()

                imageAnalysis.setAnalyzer(
                    cameraExecutor
                ) { proxy ->
                    analyzeFrame(proxy)
                }

                cameraProvider?.unbindAll()

                cameraProvider?.bindToLifecycle(
                    this,
                    CameraSelector.DEFAULT_FRONT_CAMERA,
                    preview,
                    imageAnalysis
                )

                illuminationAdequate = false
                faceLostFrames = 0

                btnCompare.isEnabled = false

                resetLivenessProgress()

                txtCameraStatus.text =
                    "Cámara frontal activa. Evaluando iluminación."

                txtChallenge.text =
                    "Coloca tu rostro frente a la cámara."

                Log.d(
                    TAG,
                    "Cámara frontal iniciada"
                )

            } catch (e: Exception) {
                Log.e(
                    TAG,
                    "Error iniciando CameraX",
                    e
                )

                txtCameraStatus.text =
                    "No se pudo iniciar la cámara: ${e.localizedMessage}"
            }

        }, ContextCompat.getMainExecutor(this))
    }

    // ---------------------------------------------------------
    // BLOQUEO DE VERIFICACIÓN FACIAL
    // ---------------------------------------------------------

    private fun getFaceLockoutStatus():
            UserDatabase.LockoutStatus? {
        return try {
            val database = UserDatabase(this)

            try {
                database.getLockoutStatus(
                    username = username,
                    face = true
                )
            } finally {
                database.close()
            }

        } catch (e: Exception) {
            Log.e(
                TAG,
                "Error consultando bloqueo facial",
                e
            )

            null
        }
    }

    private fun registerFaceFailure():
            UserDatabase.LockoutStatus? {
        return try {
            val database = UserDatabase(this)

            try {
                database.recordFailedAttempt(
                    username = username,
                    face = true
                )
            } finally {
                database.close()
            }

        } catch (e: Exception) {
            Log.e(
                TAG,
                "Error registrando intento facial",
                e
            )

            null
        }
    }

    private fun registerFaceSuccess() {
        try {
            val database = UserDatabase(this)

            try {
                database.recordSuccessfulAttempt(
                    username = username,
                    face = true
                )
            } finally {
                database.close()
            }

            Log.d(
                TAG,
                "Contador de intentos faciales reiniciado"
            )

        } catch (e: Exception) {
            Log.e(
                TAG,
                "Error reiniciando intentos faciales",
                e
            )
        }
    }

    private fun showFaceLockout(
        status: UserDatabase.LockoutStatus
    ) {
        val remaining =
            formatRemainingTime(
                status.remainingMillis
            )

        runOnUiThread {
            txtResult.text = """
                VERIFICACIÓN FACIAL BLOQUEADA

                Se alcanzó el límite de intentos fallidos.

                Tiempo restante: $remaining

                Espera a que termine el bloqueo para volver a intentarlo.
            """.trimIndent()

            txtChallenge.text =
                "La verificación facial está temporalmente bloqueada."

            txtCameraStatus.text =
                "Bloqueo de seguridad activo."

            btnCompare.isEnabled = true
        }

        Log.w(
            TAG,
            "Bloqueo facial activo. " +
                    "Tiempo restante: ${status.remainingMillis} ms"
        )
    }
    // ---------------------------------------------------------
    // ILUMINACIÓN FACIAL
    // ---------------------------------------------------------

    private fun calculateFaceBrightness(
        bitmap: Bitmap,
        detection: MediaPipeDetection
    ): IlluminationResult? {

        val bounds = detection.boundingBox

        if (
            !bounds.left.isFinite() ||
            !bounds.top.isFinite() ||
            !bounds.right.isFinite() ||
            !bounds.bottom.isFinite()
        ) return null

        val width = bounds.width()
        val height = bounds.height()

        if (width <= 1f || height <= 1f) {
            return null
        }

        val left =
            (bounds.left + width * 0.15f)
                .toInt()
                .coerceIn(0, bitmap.width - 1)

        val right =
            (bounds.right - width * 0.15f)
                .toInt()
                .coerceIn(left + 1, bitmap.width)

        val top =
            (bounds.top + height * 0.15f)
                .toInt()
                .coerceIn(0, bitmap.height - 1)

        val bottom =
            (bounds.bottom - height * 0.15f)
                .toInt()
                .coerceIn(top + 1, bitmap.height)

        if (right <= left || bottom <= top) {
            return null
        }

        val step = maxOf(
            1,
            maxOf(
                right - left,
                bottom - top
            ) / 64
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

                if (brightness < DARK_PIXEL_LIMIT) {
                    darkPixels++
                }

                if (brightness > BRIGHT_PIXEL_LIMIT) {
                    brightPixels++
                }
            }
        }

        if (pixelCount == 0) {
            return null
        }

        return IlluminationResult(
            averageBrightness =
                totalBrightness / pixelCount,

            darkPixelRatio =
                darkPixels.toDouble() / pixelCount,

            brightPixelRatio =
                brightPixels.toDouble() / pixelCount
        )
    }

    // ---------------------------------------------------------
    // INICIO DE PRUEBA DE VIDA
    // ---------------------------------------------------------

    private fun startLiveness() {
        Log.d(
            TAG,
            "ESTADO INICIAL | Reto=$isChallengeActive | " +
                    "Verificación=$isVerificationRunning | " +
                    "Finalizada=$verificationFinished | " +
                    "Iluminación=$illuminationAdequate"
        )

        if (
            isChallengeActive ||
            isVerificationRunning
        ) {
            Log.d(
                TAG,
                "Inicio ignorado: ya existe un proceso activo."
            )
            return
        }

        if (
            mediaPipeDetector == null ||
            embeddingProcessor == null ||
            blinkFaceDetector == null
        ) {
            txtResult.text =
                "Los motores biométricos no están disponibles."
            return
        }

        val cooldownRemaining =
            getRetryRemaining()

        if (cooldownRemaining > 0L) {
            txtResult.text =
                "Debes esperar ${formatRemainingTime(cooldownRemaining)} " +
                        "antes de volver a verificar."

            btnCompare.isEnabled = false
            return
        }

        val lockoutStatus =
            getFaceLockoutStatus()

        if (lockoutStatus == null) {
            txtResult.text =
                "No se pudo consultar el estado de seguridad. Intenta nuevamente."
            return
        }

        if (lockoutStatus.isLocked) {
            showFaceLockout(lockoutStatus)
            return
        }

        if (!illuminationAdequate) {
            txtResult.text =
                "Mejora la iluminación antes de iniciar."

            txtChallenge.text =
                "Busca un lugar mejor iluminado."

            btnCompare.isEnabled = false
            return
        }

        btnCompare.isEnabled = false

        try {
            cameraExecutor.execute {

                verificationFinished = false
                verificationTimedOut = false

                challengeStep = 0
                stableFrames = 0
                faceLostFrames = 0

                firstTurnDirection =
                    if (Random.nextBoolean()) 1 else -1

                randomizedActions =
                    listOf(
                        RandomAction.BLINK,
                        RandomAction.VERTICAL
                    ).shuffled()

                verticalDirection =
                    if (Random.nextBoolean()) {
                        VerticalDirection.UP
                    } else {
                        VerticalDirection.DOWN
                    }

                blinkLivenessDetector.reset()

                isChallengeActive = true
                isVerificationRunning = false

                // Iniciar límite de 30 s total y 10 s por paso
                startVerificationTimers()

                val firstMessage =
                    when (randomizedActions.first()) {
                        RandomAction.BLINK ->
                            "Mira a la cámara y parpadea cuando estés listo."

                        RandomAction.VERTICAL ->
                            if (
                                verticalDirection ==
                                VerticalDirection.UP
                            ) {
                                "Inclina ligeramente la cabeza hacia arriba."
                            } else {
                                "Inclina ligeramente la cabeza hacia abajo."
                            }
                    }

                runOnUiThread {
                    txtResult.text =
                        "Realizando prueba de vida..."

                    txtChallenge.text =
                        firstMessage

                    txtCameraStatus.text =
                        "Preparando prueba de vida."

                    txtProgress.text =
                        "PASO 1 DE 5"

                    progressLiveness.progress = 1
                }

                Log.d(
                    TAG,
                    "Prueba iniciada. Orden: $randomizedActions. " +
                            "Dirección vertical: $verticalDirection"
                )
            }

        } catch (e: Exception) {
            Log.e(
                TAG,
                "No se pudo iniciar el reto",
                e
            )

            cancelVerificationTimers()

            runOnUiThread {
                isChallengeActive = false
                verificationFinished = true

                btnCompare.isEnabled =
                    illuminationAdequate

                txtResult.text =
                    "No se pudo iniciar la verificación."
            }
        }
    }

    // ---------------------------------------------------------
    // SEÑALES FACIALES ML KIT
    // ---------------------------------------------------------

    private fun detectFaceSignals(
        bitmap: Bitmap,
        checkBlink: Boolean
    ): FaceSignals? {

        val detector =
            blinkFaceDetector ?: return null

        return try {
            val inputImage =
                InputImage.fromBitmap(bitmap, 0)

            val faces = Tasks.await(
                detector.process(inputImage),
                1200,
                TimeUnit.MILLISECONDS
            )

            val face = faces.maxByOrNull {
                it.boundingBox.width().toLong() *
                        it.boundingBox.height().toLong()
            } ?: return null

            val blinkStatus =
                if (checkBlink) {
                    blinkLivenessDetector.update(
                        face.leftEyeOpenProbability,
                        face.rightEyeOpenProbability
                    )
                } else {
                    null
                }

            val pitch = -face.headEulerAngleX
            val yaw = -face.headEulerAngleY

            Log.d(
                TAG,
                "ML KIT | Pitch: $pitch | Yaw: $yaw | " +
                        "Blink: $blinkStatus"
            )

            FaceSignals(
                blinkStatus = blinkStatus,
                pitch = pitch
            )

        } catch (e: Exception) {
            Log.e(
                TAG,
                "Error detectando señales faciales",
                e
            )
            null
        }
    }

    // ---------------------------------------------------------
    // ANALIZAR FOTOGRAMAS
    // ---------------------------------------------------------

    private fun analyzeFrame(
        imageProxy: ImageProxy
    ) {
        var bitmap: Bitmap? = null

        try {
            if (
                isVerificationRunning ||
                verificationFinished ||
                verificationTimedOut
            ) return

            val frameBitmap =
                imageProxyToBitmap(imageProxy)

            bitmap = frameBitmap

            val detector =
                mediaPipeDetector ?: return

            val detection =
                detector.detect(frameBitmap)

            if (
                detection == null ||
                detection.confidence < 0.5f ||
                detection.keypoints.size < 3
            ) {
                if (isChallengeActive) {
                    faceLostFrames++
                    stableFrames = 0

                    runOnUiThread {
                        txtCameraStatus.text =
                            "Rostro perdido momentáneamente..."

                        txtChallenge.text =
                            "Vuelve a colocar tu rostro frente a la cámara."
                    }

                    if (
                        faceLostFrames >
                        MAX_FACE_LOST_FRAMES
                    ) {
                        isChallengeActive = false
                        challengeStep = 0
                        faceLostFrames = 0
                        stableFrames = 0

                        // Cancelar temporizadores de este intento
                        cancelVerificationTimers()

                        verificationTimedOut = false
                        verificationFinished = false

                        blinkLivenessDetector.reset()

                        resetLivenessProgress()

                        runOnUiThread {
                            txtResult.text =
                                "Se perdió el rostro. Intenta nuevamente."

                            btnCompare.isEnabled =
                                illuminationAdequate
                        }
                    }

                } else {
                    illuminationAdequate = false

                    runOnUiThread {
                        txtCameraStatus.text =
                            "No se detecta un rostro. Mira hacia la cámara."

                        txtChallenge.text =
                            "Coloca tu rostro frente a la cámara."

                        txtResult.text =
                            "Esperando detección facial."

                        btnCompare.isEnabled = false
                    }
                }

                return
            }

            faceLostFrames = 0

            val lighting =
                calculateFaceBrightness(
                    frameBitmap,
                    detection
                )

            Log.d(
                TAG,
                "ILUMINACIÓN | Promedio=${lighting?.averageBrightness} " +
                        "| Oscuros=${lighting?.darkPixelRatio} " +
                        "| Brillantes=${lighting?.brightPixelRatio} " +
                        "| Adecuada=${lighting?.isAdequate} " +
                        "| Reto activo=$isChallengeActive"
            )

            if (!isChallengeActive) {
                if (lighting == null) {
                    illuminationAdequate = false
                    stableFrames = 0

                    runOnUiThread {
                        txtCameraStatus.text =
                            "No se pudo evaluar la iluminación."

                        txtChallenge.text =
                            "Ajusta la posición de tu rostro."

                        btnCompare.isEnabled = false
                    }

                    return
                }

                if (!lighting.isAdequate) {
                    illuminationAdequate = false
                    stableFrames = 0

                    val message =
                        when {
                            lighting.averageBrightness <
                                    MIN_FACE_BRIGHTNESS ->
                                "Busca un lugar con más iluminación."

                            lighting.darkPixelRatio >
                                    MAX_DARK_PIXEL_RATIO ->
                                "Hay demasiadas sombras. Ilumina mejor tu rostro."

                            lighting.brightPixelRatio >
                                    MAX_BRIGHT_PIXEL_RATIO ->
                                "Hay demasiada luz directa. Evita los reflejos."

                            else ->
                                "Ajusta la iluminación de tu rostro."
                        }

                    runOnUiThread {
                        txtCameraStatus.text =
                            "Iluminación inadecuada."

                        txtChallenge.text =
                            message

                        txtResult.text =
                            "No se puede iniciar la verificación.\n$message"

                        btnCompare.isEnabled = false
                    }

                    return
                }

                illuminationAdequate = true

                runOnUiThread {
                    txtCameraStatus.text =
                        "Iluminación adecuada."

                    if (
                        !isChallengeActive &&
                        !isVerificationRunning &&
                        resultButtons.visibility != View.VISIBLE
                    ) {
                        txtChallenge.text =
                            "Iluminación adecuada. Puedes iniciar."

                        btnCompare.isEnabled = true
                    }
                }
            }

            if (!isChallengeActive) return

            val eyes = detection.keypoints
                .take(2)
                .sortedBy { it.x }

            val leftEye = eyes[0]
            val rightEye = eyes[1]
            val nose = detection.keypoints[2]

            val eyeCenterX =
                (leftEye.x + rightEye.x) / 2f

            val eyeDistance =
                abs(rightEye.x - leftEye.x)

            if (eyeDistance < 1f) {
                stableFrames = 0
                return
            }

            val noseOffset =
                (nose.x - eyeCenterX) / eyeDistance

            val checkBlink =
                challengeStep in 0..1 &&
                        randomizedActions.getOrNull(challengeStep) ==
                        RandomAction.BLINK

            val faceSignals =
                detectFaceSignals(
                    frameBitmap,
                    checkBlink
                )

            if (faceSignals == null) {
                stableFrames = 0
                return
            }

            processLiveness(
                noseOffset,
                frameBitmap,
                faceSignals
            )

        } catch (e: Exception) {
            Log.e(
                TAG,
                "Error analizando fotograma",
                e
            )

        } finally {
            bitmap?.let {
                if (!it.isRecycled) {
                    it.recycle()
                }
            }

            imageProxy.close()
        }
    }

    // ---------------------------------------------------------
    // PROCESAMIENTO DE DESAFÍOS
    // ---------------------------------------------------------

    private fun processLiveness(
        noseOffset: Float,
        bitmap: Bitmap,
        signals: FaceSignals
    ) {
        if (
            !isChallengeActive ||
            isVerificationRunning ||
            verificationTimedOut
        ) return

        when (challengeStep) {
            0, 1 -> {
                when (randomizedActions[challengeStep]) {
                    RandomAction.BLINK -> {
                        if (
                            signals.blinkStatus ==
                            BlinkLivenessDetector.Status.COMPLETED
                        ) {
                            completeRandomAction()
                        }
                    }

                    RandomAction.VERTICAL -> {
                        val correctMovement =
                            when (verticalDirection) {
                                VerticalDirection.UP ->
                                    signals.pitch <=
                                            UP_PITCH_THRESHOLD

                                VerticalDirection.DOWN ->
                                    signals.pitch >=
                                            DOWN_PITCH_THRESHOLD
                            }

                        if (correctMovement) {
                            stableFrames++
                        } else {
                            stableFrames = 0
                        }

                        if (
                            stableFrames >=
                            REQUIRED_FRAMES
                        ) {
                            completeRandomAction()
                        }
                    }
                }
            }

            2 -> {
                val centered =
                    abs(noseOffset) <= CENTER_THRESHOLD &&
                            abs(signals.pitch) <=
                            CENTER_PITCH_THRESHOLD

                if (centered) {
                    stableFrames++
                } else {
                    stableFrames = 0
                }

                if (stableFrames >= REQUIRED_FRAMES) {
                    stableFrames = 0
                    challengeStep = 3

                    // Nuevo paso: reiniciar 10 segundos
                    resetStepTimer()

                    updateLivenessProgress(3)

                    val message =
                        if (firstTurnDirection > 0) {
                            "Gira ligeramente hacia tu izquierda."
                        } else {
                            "Gira ligeramente hacia tu derecha."
                        }

                    updateChallenge(message)
                }
            }

            3 -> {
                val correctDirection =
                    noseOffset *
                            firstTurnDirection >
                            TURN_THRESHOLD

                if (correctDirection) {
                    stableFrames++
                } else {
                    stableFrames = 0
                }

                if (stableFrames >= REQUIRED_FRAMES) {
                    stableFrames = 0
                    challengeStep = 4

                    // Nuevo paso: reiniciar 10 segundos
                    resetStepTimer()

                    updateLivenessProgress(4)

                    val message =
                        if (firstTurnDirection > 0) {
                            "Ahora gira hacia tu derecha."
                        } else {
                            "Ahora gira hacia tu izquierda."
                        }

                    updateChallenge(message)
                }
            }

            4 -> {
                val oppositeDirection =
                    noseOffset *
                            firstTurnDirection <
                            -TURN_THRESHOLD

                if (oppositeDirection) {
                    stableFrames++
                } else {
                    stableFrames = 0
                }

                if (stableFrames >= REQUIRED_FRAMES) {
                    stableFrames = 0
                    challengeStep = 5

                    // Nuevo paso: reiniciar 10 segundos
                    resetStepTimer()

                    updateLivenessProgress(5)

                    updateChallenge(
                        "Vuelve a mirar directamente a la cámara."
                    )
                }
            }

            5 -> {
                val centered =
                    abs(noseOffset) <= CENTER_THRESHOLD &&
                            abs(signals.pitch) <=
                            CENTER_PITCH_THRESHOLD

                if (centered) {
                    stableFrames++
                } else {
                    stableFrames = 0
                }

                if (stableFrames >= REQUIRED_FRAMES) {
                    stableFrames = 0
                    completeLiveness(bitmap)
                }
            }
        }
    }

    // ---------------------------------------------------------
    // COMPLETAR RETOS ALEATORIOS
    // ---------------------------------------------------------

    private fun completeRandomAction() {
        stableFrames = 0
        challengeStep++

        // Nuevo paso: reiniciar límite de 10 segundos
        resetStepTimer()

        if (challengeStep < 2) {
            updateLivenessProgress(1)

            val nextMessage =
                when (randomizedActions[challengeStep]) {
                    RandomAction.BLINK ->
                        "Ahora realiza un parpadeo."

                    RandomAction.VERTICAL ->
                        if (
                            verticalDirection ==
                            VerticalDirection.UP
                        ) {
                            "Ahora inclina ligeramente la cabeza hacia arriba."
                        } else {
                            "Ahora inclina ligeramente la cabeza hacia abajo."
                        }
                }

            updateChallenge(nextMessage)

        } else {
            challengeStep = 2

            // El primer paso de centrado también tiene 10 s
            resetStepTimer()

            updateLivenessProgress(2)

            updateChallenge(
                "Desafíos completados. Mira directamente a la cámara."
            )
        }

        Log.d(
            TAG,
            "Desafío aleatorio completado. " +
                    "Siguiente paso: $challengeStep"
        )
    }

    private fun updateChallenge(
        message: String
    ) {
        runOnUiThread {
            txtChallenge.text = message

            txtCameraStatus.text =
                "Prueba de vida en progreso..."
        }

        Log.d(
            TAG,
            "Nuevo desafío: $message"
        )
    }

    // ---------------------------------------------------------
    // FINALIZAR PRUEBA DE VIDA
    // ---------------------------------------------------------

    private fun completeLiveness(
        bitmap: Bitmap
    ) {
        if (
            isVerificationRunning ||
            verificationFinished ||
            verificationTimedOut
        ) return

        isVerificationRunning = true
        isChallengeActive = false

        // Se desactiva el límite por paso; sigue el total de 30 s
        stepDeadline = 0L

        updateLivenessProgress(5)

        val detection =
            mediaPipeDetector?.detect(bitmap)

        if (
            detection == null ||
            detection.confidence < 0.5f ||
            detection.keypoints.size < 3
        ) {
            showCaptureError(
                "No se pudo capturar el rostro para verificar."
            )

            releaseAfterCaptureError()
            return
        }

        val finalLighting =
            calculateFaceBrightness(
                bitmap,
                detection
            )

        if (
            finalLighting == null ||
            !finalLighting.isAdequate
        ) {
            illuminationAdequate = false

            val message =
                when {
                    finalLighting == null ->
                        "No se pudo evaluar la iluminación de la captura."

                    finalLighting.averageBrightness <
                            MIN_FACE_BRIGHTNESS ->
                        "La captura está demasiado oscura. Mejora la iluminación."

                    finalLighting.darkPixelRatio >
                            MAX_DARK_PIXEL_RATIO ->
                        "La captura tiene demasiadas sombras. Ilumina mejor tu rostro."

                    finalLighting.brightPixelRatio >
                            MAX_BRIGHT_PIXEL_RATIO ->
                        "La captura tiene demasiados reflejos. Evita la luz directa."

                    else ->
                        "La iluminación no es adecuada para capturar."
                }

            Log.w(
                TAG,
                "Captura rechazada por iluminación: $finalLighting"
            )

            showCaptureError(message)
            releaseAfterCaptureError()
            return
        }

        illuminationAdequate = true

        Log.d(
            TAG,
            "Iluminación final adecuada: $finalLighting"
        )

        val alignedBitmap =
            alignMediaPipeFace(
                bitmap,
                detection
            )

        if (alignedBitmap == null) {
            showCaptureError(
                "No se pudo alinear el rostro."
            )

            releaseAfterCaptureError()
            return
        }

        runOnUiThread {
            if (
                verificationTimedOut ||
                isFinishing ||
                isDestroyed
            ) return@runOnUiThread

            txtProgress.text =
                "PASO 5 DE 5"

            progressLiveness.progress = 5

            txtChallenge.text =
                "Prueba de vida completada."

            txtCameraStatus.text =
                "Rostro preparado. Verificando identidad..."

            txtResult.text =
                "Prueba de vida completada.\n" +
                        "Comparando plantillas faciales..."
        }

        try {
            faceExecutor.execute {
                verifyCapturedFace(alignedBitmap)
            }

        } catch (e: Exception) {
            if (!alignedBitmap.isRecycled) {
                alignedBitmap.recycle()
            }

            Log.e(
                TAG,
                "No se pudo iniciar la comparación",
                e
            )

            showCaptureError(
                "No se pudo iniciar la comparación facial."
            )

            releaseAfterCaptureError()
        }
    }

    // ---------------------------------------------------------
    // LIBERAR PROCESO DESPUÉS DE ERROR DE CAPTURA
    // ---------------------------------------------------------

    private fun releaseAfterCaptureError() {
        isVerificationRunning = false

        cancelVerificationTimers()

        runOnUiThread {
            if (
                !isFinishing &&
                !isDestroyed &&
                verificationFinished
            ) {
                btnCompare.isEnabled =
                    illuminationAdequate
            }

            Log.d(
                TAG,
                "Bloqueo de captura liberado. " +
                        "Iluminación adecuada: $illuminationAdequate"
            )
        }
    }

    // ---------------------------------------------------------
    // ERRORES DE CAPTURA
    // ---------------------------------------------------------

    private fun showCaptureError(
        message: String
    ) {
        verificationFinished = true

        cancelVerificationTimers()

        runOnUiThread {
            if (
                verificationTimedOut ||
                isFinishing ||
                isDestroyed
            ) return@runOnUiThread

            txtResult.text = message

            txtChallenge.text =
                "Puedes volver a intentarlo."

            txtCameraStatus.text =
                "Captura no completada."

            txtProgress.text =
                "PASO 1 DE 5"

            progressLiveness.progress = 0

            showResultButtons(
                accepted = false,
                technicalError = true
            )
        }
    }

    // ---------------------------------------------------------
    // VERIFICACIÓN DE IDENTIDAD
    // ---------------------------------------------------------

    private fun verifyCapturedFace(
        bitmap: Bitmap
    ) {
        try {
            if (verificationTimedOut) return

            val database = UserDatabase(this)

            val primaryTemplates:
                    Triple<FloatArray, FloatArray, FloatArray>?

            val additionalTemplates:
                    Triple<FloatArray, FloatArray, FloatArray>?

            try {
                primaryTemplates =
                    database.getFaceEmbeddings(username)

                additionalTemplates =
                    database.getNoGlassesFaceEmbeddings(username)

            } finally {
                database.close()
            }

            if (verificationTimedOut) return

            if (primaryTemplates == null) {
                throw IllegalStateException(
                    "No se encontraron plantillas faciales principales para esta cuenta."
                )
            }

            val processor =
                embeddingProcessor
                    ?: throw IllegalStateException(
                        "MobileFaceNet no está inicializado."
                    )

            val newEmbedding =
                processor.generateEmbedding(bitmap)

            if (verificationTimedOut) return

            require(
                newEmbedding.size == 128 &&
                        newEmbedding.all { it.isFinite() } &&
                        newEmbedding.any { it != 0f }
            ) {
                "El embedding capturado no es válido."
            }

            val registeredEmbeddings =
                mutableListOf<Pair<String, FloatArray>>()

            registeredEmbeddings.add(
                "Principal frontal" to
                        primaryTemplates.first
            )

            registeredEmbeddings.add(
                "Principal izquierda" to
                        primaryTemplates.second
            )

            registeredEmbeddings.add(
                "Principal derecha" to
                        primaryTemplates.third
            )

            if (additionalTemplates != null) {
                registeredEmbeddings.add(
                    "Complementaria frontal" to
                            additionalTemplates.first
                )

                registeredEmbeddings.add(
                    "Complementaria izquierda" to
                            additionalTemplates.second
                )

                registeredEmbeddings.add(
                    "Complementaria derecha" to
                            additionalTemplates.third
                )

                Log.d(
                    TAG,
                    "Se encontraron plantillas principales y complementarias."
                )

            } else {
                Log.d(
                    TAG,
                    "El usuario no tiene plantillas complementarias. " +
                            "Se utilizarán únicamente las principales."
                )
            }

            registeredEmbeddings.forEach {
                    (name, embedding) ->

                require(
                    embedding.size == 128 &&
                            embedding.all { value ->
                                value.isFinite()
                            } &&
                            embedding.any { value ->
                                value != 0f
                            }
                ) {
                    "La plantilla '$name' no es válida."
                }
            }

            if (verificationTimedOut) return

            val similarities =
                registeredEmbeddings.map {
                        (name, template) ->

                    val similarity =
                        FaceSimilarity.cosineSimilarity(
                            newEmbedding,
                            template
                        )

                    require(similarity.isFinite()) {
                        "Similitud inválida con la plantilla '$name'."
                    }

                    Log.d(
                        TAG,
                        "Similitud con $name: $similarity"
                    )

                    name to similarity
                }

            val bestMatch =
                similarities.maxByOrNull {
                    it.second
                } ?: throw IllegalStateException(
                    "No se pudieron comparar las plantillas."
                )

            val bestTemplateName =
                bestMatch.first

            val bestSimilarity =
                bestMatch.second

            val accepted =
                bestSimilarity >= THRESHOLD

            val formattedSimilarity =
                String.format(
                    Locale.US,
                    "%.4f",
                    bestSimilarity
                )

            // Evitar registrar intentos si ya expiró
            if (verificationTimedOut) {
                Log.w(
                    TAG,
                    "Resultado descartado: el tiempo expiró."
                )
                return
            }

            val lockoutAfterAttempt =
                if (accepted) {
                    registerFaceSuccess()
                    null
                } else {
                    registerFaceFailure()
                }

            runOnUiThread {
                // Evitar resultados tardíos en UI
                if (
                    verificationTimedOut ||
                    isFinishing ||
                    isDestroyed
                ) {
                    return@runOnUiThread
                }

                verificationFinished = true
                cancelVerificationTimers()

                if (accepted) {
                    txtResult.text = """
                        IDENTIDAD VALIDADA

                        Usuario: $username

                        Similitud máxima:
                        $formattedSimilarity

                        Plantilla coincidente:
                        $bestTemplateName

                        Umbral: $THRESHOLD

                        Prueba de vida completada.
                    """.trimIndent()

                } else {
                    val failureMessage =
                        when {
                            lockoutAfterAttempt == null ->
                                """
                                IDENTIDAD NO VALIDADA

                                Usuario: $username

                                Similitud máxima: $formattedSimilarity

                                Umbral: $THRESHOLD

                                No se pudo actualizar el registro de intentos.
                                """.trimIndent()

                            lockoutAfterAttempt.isLocked ->
                                """
                                IDENTIDAD NO VALIDADA

                                Usuario: $username

                                Similitud máxima: $formattedSimilarity

                                Umbral: $THRESHOLD

                                Se alcanzó el límite de intentos.

                                Bloqueo temporal: ${
                                    formatRemainingTime(
                                        lockoutAfterAttempt.remainingMillis
                                    )
                                }
                                """.trimIndent()

                            else -> {
                                val remainingAttempts =
                                    (
                                            MAX_FACE_FAILED_ATTEMPTS -
                                                    lockoutAfterAttempt.failedAttempts
                                            ).coerceAtLeast(0)

                                """
                                IDENTIDAD NO VALIDADA

                                Usuario: $username

                                Similitud máxima: $formattedSimilarity

                                Umbral: $THRESHOLD

                                Intentos fallidos: ${lockoutAfterAttempt.failedAttempts}

                                Intentos restantes: $remainingAttempts
                                """.trimIndent()
                            }
                        }

                    txtResult.text =
                        failureMessage
                }

                txtCameraStatus.text =
                    "Verificación finalizada."

                txtProgress.text =
                    "PASO 5 DE 5"

                progressLiveness.progress = 5

                if (accepted) {
                    txtChallenge.text =
                        "Identidad validada correctamente."

                    showResultButtons(
                        accepted = true
                    )

                } else {
                    if (
                        lockoutAfterAttempt != null &&
                        lockoutAfterAttempt.isLocked
                    ) {
                        txtChallenge.text =
                            "Se alcanzó el límite de intentos. " +
                                    "Espera a que termine el bloqueo."

                    } else {
                        txtChallenge.text =
                            "Identidad no validada. " +
                                    "Puedes volver a intentarlo."
                    }

                    showResultButtons(
                        accepted = false,
                        technicalError = lockoutAfterAttempt == null,
                        lockoutStatus = lockoutAfterAttempt
                    )
                }
            }

            Log.d(
                TAG,
                "Mejor coincidencia: $bestTemplateName | " +
                        "Similitud: $bestSimilarity | " +
                        "Umbral: $THRESHOLD | " +
                        "Identidad aceptada: $accepted"
            )

        } catch (e: Exception) {
            Log.e(
                TAG,
                "Error verificando identidad",
                e
            )

            runOnUiThread {
                if (
                    verificationTimedOut ||
                    isFinishing ||
                    isDestroyed
                ) {
                    return@runOnUiThread
                }

                verificationFinished = true
                cancelVerificationTimers()

                txtResult.text =
                    "Error durante la verificación: " +
                            "${e.localizedMessage}"

                txtChallenge.text =
                    "Puedes volver a intentarlo."

                txtCameraStatus.text =
                    "Verificación no completada."

                txtProgress.text =
                    "PASO 1 DE 5"

                progressLiveness.progress = 0

                showResultButtons(
                    accepted = false,
                    technicalError = true
                )
            }

        } finally {
            if (!bitmap.isRecycled) {
                bitmap.recycle()
            }

            isVerificationRunning = false

            runOnUiThread {
                if (
                    !isFinishing &&
                    !isDestroyed
                ) {
                    Log.d(
                        TAG,
                        "Verificación liberada. " +
                                "Resultado permanece en pantalla."
                    )
                }
            }
        }
    }

    // ---------------------------------------------------------
    // CONVERSIÓN IMAGEPROXY A BITMAP
    // ---------------------------------------------------------

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

        // Copiar plano Y
        for (row in 0 until height) {
            for (col in 0 until width) {
                val index =
                    yBase +
                            row * yPlane.rowStride +
                            col * yPlane.pixelStride

                nv21[outputIndex++] = yBuffer.get(index)
            }
        }

        // Copiar planos de color en formato NV21 (V, U)
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

        val rotation =
            imageProxy.imageInfo.rotationDegrees

        if (rotation == 0) {
            return original
        }

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

    // ---------------------------------------------------------
    // ALINEACIÓN FACIAL
    // ---------------------------------------------------------

    private fun alignMediaPipeFace(
        bitmap: Bitmap,
        detection: MediaPipeDetection
    ): Bitmap? {
        if (detection.keypoints.size < 3) {
            return null
        }

        val eyes =
            detection.keypoints
                .take(2)
                .sortedBy { it.x }

        val leftEye: PointF =
            eyes[0]

        val rightEye: PointF =
            eyes[1]

        val nose: PointF =
            detection.keypoints[2]

        val sourcePoints =
            floatArrayOf(
                leftEye.x, leftEye.y,
                rightEye.x, rightEye.y,
                nose.x, nose.y
            )

        val targetPoints =
            floatArrayOf(
                38.3f, 51.7f,
                73.5f, 51.5f,
                56.0f, 71.7f
            )

        val matrix = Matrix()

        if (
            !matrix.setPolyToPoly(
                sourcePoints,
                0,
                targetPoints,
                0,
                3
            )
        ) {
            Log.e(
                TAG,
                "Error calculando matriz de alineación"
            )

            return null
        }

        val alignedBitmap =
            Bitmap.createBitmap(
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

    // ---------------------------------------------------------
    // CIERRE DE RECURSOS
    // ---------------------------------------------------------

    override fun onDestroy() {
        cancelVerificationTimers()

        retryHandler.removeCallbacks(
            retryCountdownRunnable
        )

        isChallengeActive = false
        isVerificationRunning = false
        verificationFinished = false

        cameraProvider?.unbindAll()
        cameraProvider = null

        if (!cameraExecutor.isShutdown) {
            cameraExecutor.execute {
                try {
                    mediaPipeDetector?.close()
                    mediaPipeDetector = null

                    blinkFaceDetector?.close()
                    blinkFaceDetector = null

                } catch (e: Exception) {
                    Log.e(
                        TAG,
                        "Error cerrando detectores",
                        e
                    )
                }
            }

            cameraExecutor.shutdown()
        }

        if (!faceExecutor.isShutdown) {
            faceExecutor.execute {
                try {
                    embeddingProcessor?.close()
                    embeddingProcessor = null

                } catch (e: Exception) {
                    Log.e(
                        TAG,
                        "Error cerrando MobileFaceNet",
                        e
                    )
                }
            }

            faceExecutor.shutdown()
        }

        super.onDestroy()
    }
}
