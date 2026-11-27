package com.example.myapplication

import android.content.Intent
import android.os.Bundle
import android.widget.Button
import android.widget.EditText
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import kotlin.math.ceil

class LoginActivity : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_login)

        val username = findViewById<EditText>(R.id.etUsername)
        val password = findViewById<EditText>(R.id.etPassword)

        val btnLogin = findViewById<Button>(R.id.btnLogin)
        val btnRegister = findViewById<Button>(R.id.btnGoRegister)

        btnLogin.setOnClickListener {

            val user = username.text.toString().trim()
            val pass = password.text.toString()

            if (user.isEmpty() || pass.isEmpty()) {
                Toast.makeText(
                    this,
                    "Completa todos los campos",
                    Toast.LENGTH_SHORT
                ).show()

                return@setOnClickListener
            }

            val database = UserDatabase(this)

            try {

                // Comprobar bloqueo del inicio de sesión
                val lockout = database.getLockoutStatus(
                    user,
                    false
                )

                if (lockout.isLocked) {

                    val seconds = ceil(
                        lockout.remainingMillis / 1000.0
                    ).toInt()

                    Toast.makeText(
                        this,
                        "Inicio de sesión bloqueado. Intenta en $seconds segundos.",
                        Toast.LENGTH_LONG
                    ).show()

                    return@setOnClickListener
                }

                // Validar credenciales
                val valid = database.authenticate(user, pass)

                if (valid) {

                    // Reiniciar intentos fallidos
                    database.recordSuccessfulAttempt(user, false)

                    Toast.makeText(
                        this,
                        "¡Inicio de sesión correcto!",
                        Toast.LENGTH_SHORT
                    ).show()

                    // Ir al menú principal
                    val intent = Intent(
                        this,
                        UserMenuActivity::class.java
                    )

                    intent.putExtra("username", user)

                    startActivity(intent)
                    finish()

                } else {

                    // Registrar intento fallido
                    val result = database.recordFailedAttempt(
                        user,
                        false
                    )

                    if (result.isLocked) {

                        val seconds = ceil(
                            result.remainingMillis / 1000.0
                        ).toInt()

                        Toast.makeText(
                            this,
                            "Has alcanzado 3 intentos. Inicio bloqueado durante $seconds segundos.",
                            Toast.LENGTH_LONG
                        ).show()

                    } else {

                        val remaining =
                            3 - result.failedAttempts

                        Toast.makeText(
                            this,
                            "Usuario o contraseña incorrectos. Intentos restantes: $remaining",
                            Toast.LENGTH_LONG
                        ).show()
                    }
                }

            } finally {
                database.close()
            }
        }

        // Ir al registro
        btnRegister.setOnClickListener {

            startActivity(
                Intent(
                    this,
                    RegisterActivity::class.java
                )
            )
        }
    }
}