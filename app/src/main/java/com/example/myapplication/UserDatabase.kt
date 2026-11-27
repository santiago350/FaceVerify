
package com.example.myapplication

import android.content.ContentValues
import android.content.Context
import android.database.sqlite.SQLiteDatabase
import android.database.sqlite.SQLiteOpenHelper
import android.database.sqlite.SQLiteConstraintException
import android.security.keystore.KeyGenParameterSpec
import android.security.keystore.KeyProperties
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.security.KeyStore
import java.security.MessageDigest
import java.security.SecureRandom
import javax.crypto.Cipher
import javax.crypto.KeyGenerator
import javax.crypto.SecretKey
import javax.crypto.SecretKeyFactory
import javax.crypto.spec.GCMParameterSpec
import javax.crypto.spec.PBEKeySpec

class UserDatabase(context: Context) :
    SQLiteOpenHelper(context, "FaceVerify.db", null, 4) {

    companion object {
        private const val ITERATIONS = 120000
        private const val KEY_LENGTH = 256
        private const val SALT_LENGTH = 16
        private const val EMBEDDING_SIZE = 128

        private const val KEY_ALIAS = "FaceVerifyEmbeddingKey"
        private const val ANDROID_KEYSTORE = "AndroidKeyStore"

        private const val MAX_FAILED_ATTEMPTS = 3
        private const val INITIAL_LOCKOUT_MS = 15_000L
    }

    data class LockoutStatus(
        val isLocked: Boolean,
        val remainingMillis: Long,
        val failedAttempts: Int,
        val lockoutLevel: Int
    )

    override fun onCreate(db: SQLiteDatabase) {

        db.execSQL(
            """
            CREATE TABLE users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                full_name TEXT NOT NULL,
                username TEXT NOT NULL COLLATE NOCASE UNIQUE,
                password_salt BLOB NOT NULL,
                password_hash BLOB NOT NULL
            )
            """.trimIndent()
        )

        createFaceTemplatesTable(db)
        createNoGlassesTemplatesTable(db)
        createSecurityTable(db)
    }

    private fun createFaceTemplatesTable(db: SQLiteDatabase) {

        db.execSQL(
            """
            CREATE TABLE IF NOT EXISTS face_templates (
                username TEXT PRIMARY KEY COLLATE NOCASE,
                front_embedding BLOB NOT NULL,
                left_embedding BLOB NOT NULL,
                right_embedding BLOB NOT NULL
            )
            """.trimIndent()
        )
    }

    private fun createNoGlassesTemplatesTable(db: SQLiteDatabase) {

        db.execSQL(
            """
            CREATE TABLE IF NOT EXISTS face_templates_no_glasses (
                username TEXT PRIMARY KEY COLLATE NOCASE,
                front_embedding BLOB NOT NULL,
                left_embedding BLOB NOT NULL,
                right_embedding BLOB NOT NULL
            )
            """.trimIndent()
        )
    }

    // Tabla independiente para los intentos y bloqueos.
    private fun createSecurityTable(db: SQLiteDatabase) {

        db.execSQL(
            """
            CREATE TABLE IF NOT EXISTS security_attempts (
                username TEXT PRIMARY KEY COLLATE NOCASE,

                password_failures INTEGER NOT NULL DEFAULT 0,
                password_lock_until INTEGER NOT NULL DEFAULT 0,
                password_lock_level INTEGER NOT NULL DEFAULT 0,

                face_failures INTEGER NOT NULL DEFAULT 0,
                face_lock_until INTEGER NOT NULL DEFAULT 0,
                face_lock_level INTEGER NOT NULL DEFAULT 0
            )
            """.trimIndent()
        )
    }

    override fun onUpgrade(
        db: SQLiteDatabase,
        oldVersion: Int,
        newVersion: Int
    ) {

        if (oldVersion < 2) {
            createFaceTemplatesTable(db)
        }

        if (oldVersion < 3) {
            createNoGlassesTemplatesTable(db)
        }

        if (oldVersion < 4) {
            createSecurityTable(db)
        }
    }

    // Validar dimensiones, valores y magnitud del embedding.
    private fun isValidEmbedding(embedding: FloatArray): Boolean {

        if (embedding.size != EMBEDDING_SIZE) return false

        var normSquared = 0.0

        for (value in embedding) {

            if (!value.isFinite()) return false

            normSquared += value.toDouble() * value.toDouble()
        }

        return normSquared > 1e-12
    }

    // Crear usuario.
    fun createUser(
        fullName: String,
        username: String,
        password: String
    ): Boolean {

        val salt = ByteArray(SALT_LENGTH)
        SecureRandom().nextBytes(salt)

        val passwordHash = hashPassword(password, salt)

        val values = ContentValues().apply {
            put("full_name", fullName)
            put("username", username)
            put("password_salt", salt)
            put("password_hash", passwordHash)
        }

        return try {

            val result = writableDatabase.insertOrThrow(
                "users",
                null,
                values
            )

            if (result != -1L) {

                // Inicializar la seguridad de la cuenta nueva.
                val securityValues = ContentValues().apply {
                    put("username", username.trim())
                    put("password_failures", 0)
                    put("password_lock_until", 0L)
                    put("password_lock_level", 0)
                    put("face_failures", 0)
                    put("face_lock_until", 0L)
                    put("face_lock_level", 0)
                }

                writableDatabase.insertWithOnConflict(
                    "security_attempts",
                    null,
                    securityValues,
                    SQLiteDatabase.CONFLICT_REPLACE
                )

                true

            } else {
                false
            }

        } catch (e: SQLiteConstraintException) {
            false
        }
    }

    // Autenticar usuario.
    fun authenticate(
        username: String,
        password: String
    ): Boolean {

        val cursor = readableDatabase.query(
            "users",
            arrayOf("password_salt", "password_hash"),
            "username = ?",
            arrayOf(username),
            null,
            null,
            null
        )

        cursor.use {

            if (!it.moveToFirst()) {
                return false
            }

            val salt = it.getBlob(
                it.getColumnIndexOrThrow("password_salt")
            )

            val savedHash = it.getBlob(
                it.getColumnIndexOrThrow("password_hash")
            )

            val enteredHash = hashPassword(password, salt)

            return MessageDigest.isEqual(savedHash, enteredHash)
        }
    }

    // Guardar la apariencia principal.
    fun saveFaceEmbeddings(
        username: String,
        front: FloatArray,
        left: FloatArray,
        right: FloatArray
    ): Boolean {

        require(
            isValidEmbedding(front) &&
                    isValidEmbedding(left) &&
                    isValidEmbedding(right)
        ) {
            "Los embeddings deben contener 128 valores válidos y no nulos."
        }

        val userCursor = readableDatabase.query(
            "users",
            arrayOf("username"),
            "username = ?",
            arrayOf(username),
            null,
            null,
            null
        )

        val userExists = userCursor.use {
            it.moveToFirst()
        }

        if (!userExists) return false

        val values = ContentValues().apply {
            put("username", username)
            put("front_embedding", encryptEmbedding(front))
            put("left_embedding", encryptEmbedding(left))
            put("right_embedding", encryptEmbedding(right))
        }

        val result = writableDatabase.insertWithOnConflict(
            "face_templates",
            null,
            values,
            SQLiteDatabase.CONFLICT_REPLACE
        )

        return result != -1L
    }

    // Guardar la segunda apariencia: sin gafas.
    fun saveNoGlassesFaceEmbeddings(
        username: String,
        front: FloatArray,
        left: FloatArray,
        right: FloatArray
    ): Boolean {

        require(
            isValidEmbedding(front) &&
                    isValidEmbedding(left) &&
                    isValidEmbedding(right)
        ) {
            "Los embeddings deben contener 128 valores válidos y no nulos."
        }

        val userCursor = readableDatabase.query(
            "users",
            arrayOf("username"),
            "username = ?",
            arrayOf(username),
            null,
            null,
            null
        )

        val userExists = userCursor.use {
            it.moveToFirst()
        }

        if (!userExists) return false

        val values = ContentValues().apply {
            put("username", username)
            put("front_embedding", encryptEmbedding(front))
            put("left_embedding", encryptEmbedding(left))
            put("right_embedding", encryptEmbedding(right))
        }

        val result = writableDatabase.insertWithOnConflict(
            "face_templates_no_glasses",
            null,
            values,
            SQLiteDatabase.CONFLICT_REPLACE
        )

        return result != -1L
    }

    // Recuperar los embeddings principales.
    fun getFaceEmbeddings(
        username: String
    ): Triple<FloatArray, FloatArray, FloatArray>? {

        val cursor = readableDatabase.query(
            "face_templates",
            arrayOf(
                "front_embedding",
                "left_embedding",
                "right_embedding"
            ),
            "username = ?",
            arrayOf(username),
            null,
            null,
            null
        )

        return cursor.use {

            if (!it.moveToFirst()) return null

            val front = decryptEmbedding(
                it.getBlob(
                    it.getColumnIndexOrThrow("front_embedding")
                )
            )

            val left = decryptEmbedding(
                it.getBlob(
                    it.getColumnIndexOrThrow("left_embedding")
                )
            )

            val right = decryptEmbedding(
                it.getBlob(
                    it.getColumnIndexOrThrow("right_embedding")
                )
            )

            Triple(front, left, right)
        }
    }

    // Recuperar los embeddings complementarios.
    fun getNoGlassesFaceEmbeddings(
        username: String
    ): Triple<FloatArray, FloatArray, FloatArray>? {

        val cursor = readableDatabase.query(
            "face_templates_no_glasses",
            arrayOf(
                "front_embedding",
                "left_embedding",
                "right_embedding"
            ),
            "username = ?",
            arrayOf(username),
            null,
            null,
            null
        )

        return cursor.use {

            if (!it.moveToFirst()) return null

            val front = decryptEmbedding(
                it.getBlob(
                    it.getColumnIndexOrThrow("front_embedding")
                )
            )

            val left = decryptEmbedding(
                it.getBlob(
                    it.getColumnIndexOrThrow("left_embedding")
                )
            )

            val right = decryptEmbedding(
                it.getBlob(
                    it.getColumnIndexOrThrow("right_embedding")
                )
            )

            Triple(front, left, right)
        }
    }

    // Comprobar si existe la segunda apariencia.
    fun hasNoGlassesFaceTemplates(username: String): Boolean {

        val cursor = readableDatabase.query(
            "face_templates_no_glasses",
            arrayOf("username"),
            "username = ?",
            arrayOf(username),
            null,
            null,
            null
        )

        return cursor.use {
            it.moveToFirst()
        }
    }

    // Cifrar embedding.
    private fun encryptEmbedding(
        embedding: FloatArray
    ): ByteArray {

        val buffer = ByteBuffer.allocate(embedding.size * 4)
            .order(ByteOrder.BIG_ENDIAN)

        embedding.forEach {
            buffer.putFloat(it)
        }

        val cipher = Cipher.getInstance("AES/GCM/NoPadding")

        cipher.init(
            Cipher.ENCRYPT_MODE,
            getOrCreateEncryptionKey()
        )

        val encrypted = cipher.doFinal(buffer.array())
        val iv = cipher.iv

        return ByteBuffer.allocate(iv.size + encrypted.size)
            .put(iv)
            .put(encrypted)
            .array()
    }

    // Descifrar embedding.
    private fun decryptEmbedding(
        encryptedData: ByteArray
    ): FloatArray {

        val ivSize = 12

        require(encryptedData.size > ivSize) {
            "Datos cifrados inválidos."
        }

        val iv = encryptedData.copyOfRange(0, ivSize)

        val encrypted = encryptedData.copyOfRange(
            ivSize,
            encryptedData.size
        )

        val cipher = Cipher.getInstance("AES/GCM/NoPadding")

        cipher.init(
            Cipher.DECRYPT_MODE,
            getExistingEncryptionKey(),
            GCMParameterSpec(128, iv)
        )

        val decrypted = cipher.doFinal(encrypted)

        require(decrypted.size == EMBEDDING_SIZE * 4) {
            "Tamaño de embedding inválido."
        }

        val buffer = ByteBuffer.wrap(decrypted)
            .order(ByteOrder.BIG_ENDIAN)

        val embedding = FloatArray(EMBEDDING_SIZE) {
            buffer.getFloat()
        }

        require(isValidEmbedding(embedding)) {
            "El embedding recuperado no es válido."
        }

        return embedding
    }

    // Recuperar clave existente.
    private fun getExistingEncryptionKey(): SecretKey {

        val keyStore = KeyStore.getInstance(ANDROID_KEYSTORE)
        keyStore.load(null)

        return keyStore.getKey(KEY_ALIAS, null) as? SecretKey
            ?: throw IllegalStateException(
                "No existe la clave de cifrado original."
            )
    }

    // Crear clave cuando sea necesario.
    private fun getOrCreateEncryptionKey(): SecretKey {

        val keyStore = KeyStore.getInstance(ANDROID_KEYSTORE)
        keyStore.load(null)

        val existingKey = keyStore.getKey(KEY_ALIAS, null)

        if (existingKey is SecretKey) {
            return existingKey
        }

        val generator = KeyGenerator.getInstance(
            KeyProperties.KEY_ALGORITHM_AES,
            ANDROID_KEYSTORE
        )

        val spec = KeyGenParameterSpec.Builder(
            KEY_ALIAS,
            KeyProperties.PURPOSE_ENCRYPT or
                    KeyProperties.PURPOSE_DECRYPT
        )
            .setBlockModes(KeyProperties.BLOCK_MODE_GCM)
            .setEncryptionPaddings(
                KeyProperties.ENCRYPTION_PADDING_NONE
            )
            .setRandomizedEncryptionRequired(true)
            .setKeySize(256)
            .build()

        generator.init(spec)

        return generator.generateKey()
    }

    // Derivación segura de contraseña.
    private fun hashPassword(
        password: String,
        salt: ByteArray
    ): ByteArray {

        val spec = PBEKeySpec(
            password.toCharArray(),
            salt,
            ITERATIONS,
            KEY_LENGTH
        )

        return try {

            SecretKeyFactory
                .getInstance("PBKDF2WithHmacSHA256")
                .generateSecret(spec)
                .encoded

        } finally {
            spec.clearPassword()
        }
    }

    // Validar plantillas principales.
    fun validateStoredFaceEmbeddings(username: String): Boolean {

        return try {

            val embeddings = getFaceEmbeddings(username)
                ?: return false

            isValidEmbedding(embeddings.first) &&
                    isValidEmbedding(embeddings.second) &&
                    isValidEmbedding(embeddings.third)

        } catch (e: Exception) {

            android.util.Log.e(
                "FACEVERIFY",
                "Error validando plantillas principales",
                e
            )

            false
        }
    }

    // Validar plantillas complementarias.
    fun validateStoredNoGlassesFaceEmbeddings(
        username: String
    ): Boolean {

        return try {

            val embeddings = getNoGlassesFaceEmbeddings(username)
                ?: return false

            isValidEmbedding(embeddings.first) &&
                    isValidEmbedding(embeddings.second) &&
                    isValidEmbedding(embeddings.third)

        } catch (e: Exception) {

            android.util.Log.e(
                "FACEVERIFY",
                "Error validando plantillas complementarias",
                e
            )

            false
        }
    }

    // CONSULTAR ESTADO DE BLOQUEO.
    // face = false: contraseña.
    // face = true: reconocimiento facial.
    fun getLockoutStatus(
        username: String,
        face: Boolean
    ): LockoutStatus {

        val db = writableDatabase
        val normalizedUsername = username.trim()

        db.execSQL(
            "INSERT OR IGNORE INTO security_attempts (username) VALUES (?)",
            arrayOf(normalizedUsername)
        )

        val failuresColumn =
            if (face) "face_failures" else "password_failures"

        val untilColumn =
            if (face) "face_lock_until" else "password_lock_until"

        val levelColumn =
            if (face) "face_lock_level" else "password_lock_level"

        var failures = 0
        var lockUntil = 0L
        var level = 0

        val cursor = db.query(
            "security_attempts",
            arrayOf(
                failuresColumn,
                untilColumn,
                levelColumn
            ),
            "username = ?",
            arrayOf(normalizedUsername),
            null,
            null,
            null
        )

        cursor.use {

            if (it.moveToFirst()) {
                failures = it.getInt(0)
                lockUntil = it.getLong(1)
                level = it.getInt(2)
            }
        }

        val now = System.currentTimeMillis()

        // Si el bloqueo terminó, reiniciar intentos.
        // Se conserva el nivel para duplicar el siguiente bloqueo.
        if (lockUntil > 0L && now >= lockUntil) {

            val values = ContentValues().apply {
                put(failuresColumn, 0)
                put(untilColumn, 0L)
            }

            db.update(
                "security_attempts",
                values,
                "username = ?",
                arrayOf(normalizedUsername)
            )

            failures = 0
            lockUntil = 0L
        }

        val remaining = (lockUntil - now).coerceAtLeast(0L)

        return LockoutStatus(
            isLocked = remaining > 0L,
            remainingMillis = remaining,
            failedAttempts = failures,
            lockoutLevel = level
        )
    }

    // REGISTRAR INTENTO FALLIDO.
    fun recordFailedAttempt(
        username: String,
        face: Boolean
    ): LockoutStatus {

        val db = writableDatabase
        val normalizedUsername = username.trim()

        val current = getLockoutStatus(
            normalizedUsername,
            face
        )

        if (current.isLocked) return current

        val failuresColumn =
            if (face) "face_failures" else "password_failures"

        val untilColumn =
            if (face) "face_lock_until" else "password_lock_until"

        val levelColumn =
            if (face) "face_lock_level" else "password_lock_level"

        val newFailures = current.failedAttempts + 1

        val values = ContentValues()

        if (newFailures >= MAX_FAILED_ATTEMPTS) {

            val newLevel = current.lockoutLevel + 1

            val multiplier = 1L shl
                    (newLevel - 1).coerceIn(0, 20)

            val duration = INITIAL_LOCKOUT_MS * multiplier

            val lockUntil =
                System.currentTimeMillis() + duration

            values.put(failuresColumn, 0)
            values.put(untilColumn, lockUntil)
            values.put(levelColumn, newLevel)

        } else {

            values.put(failuresColumn, newFailures)
        }

        db.update(
            "security_attempts",
            values,
            "username = ?",
            arrayOf(normalizedUsername)
        )

        return getLockoutStatus(
            normalizedUsername,
            face
        )
    }

    // ELIMINAR CUENTA Y TODOS SUS DATOS ASOCIADOS.
    fun deleteUserAccount(username: String): Boolean {

        val db = writableDatabase
        val normalizedUsername = username.trim()

        return try {

            db.beginTransaction()

            // Eliminar plantillas faciales principales
            db.delete(
                "face_templates",
                "username = ?",
                arrayOf(normalizedUsername)
            )

            // Eliminar plantillas complementarias
            db.delete(
                "face_templates_no_glasses",
                "username = ?",
                arrayOf(normalizedUsername)
            )

            // Eliminar intentos fallidos y bloqueos
            db.delete(
                "security_attempts",
                "username = ?",
                arrayOf(normalizedUsername)
            )

            // Eliminar la cuenta
            val deletedUser = db.delete(
                "users",
                "username = ?",
                arrayOf(normalizedUsername)
            )

            // Confirmar únicamente si se eliminó la cuenta
            if (deletedUser > 0) {
                db.setTransactionSuccessful()
            }

            deletedUser > 0

        } catch (e: Exception) {

            android.util.Log.e(
                "FACEVERIFY",
                "Error eliminando cuenta",
                e
            )

            false

        } finally {

            db.endTransaction()
        }
    }

    // REINICIAR MECANISMO DESPUÉS DE UNA AUTENTICACIÓN EXITOSA.
    fun recordSuccessfulAttempt(
        username: String,
        face: Boolean
    ) {

        val db = writableDatabase
        val normalizedUsername = username.trim()

        db.execSQL(
            "INSERT OR IGNORE INTO security_attempts (username) VALUES (?)",
            arrayOf(normalizedUsername)
        )

        val values = ContentValues()

        if (face) {

            values.put("face_failures", 0)
            values.put("face_lock_until", 0L)
            values.put("face_lock_level", 0)

        } else {

            values.put("password_failures", 0)
            values.put("password_lock_until", 0L)
            values.put("password_lock_level", 0)
        }

        db.update(
            "security_attempts",
            values,
            "username = ?",
            arrayOf(normalizedUsername)
        )
    }
}
