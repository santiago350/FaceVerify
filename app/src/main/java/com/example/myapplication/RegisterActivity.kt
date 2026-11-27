package com.example.myapplication

import android.os.Bundle
import android.widget.Button
import android.widget.EditText
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity

class RegisterActivity : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        setContentView(R.layout.activity_register)

        val fullName = findViewById<EditText>(R.id.etFullName)
        val username = findViewById<EditText>(R.id.etRegisterUsername)
        val password = findViewById<EditText>(R.id.etRegisterPassword)
        val confirmPassword = findViewById<EditText>(R.id.etConfirmPassword)

        val btnCreate = findViewById<Button>(R.id.btnCreateAccount)
        val btnBack = findViewById<Button>(R.id.btnBackLogin)

        btnCreate.setOnClickListener {

            val name = fullName.text.toString().trim()
            val user = username.text.toString().trim()
            val pass = password.text.toString()
            val confirm = confirmPassword.text.toString()

            when {
                name.isEmpty() || user.isEmpty() ||
                        pass.isEmpty() || confirm.isEmpty() -> {

                    showMessage("Completa todos los campos")
                }

                pass.length < 8 -> {
                    showMessage("La contraseña debe tener al menos 8 caracteres")
                }

                pass != confirm -> {
                    showMessage("Las contraseñas no coinciden")
                }

                else -> {
                    val database = UserDatabase(this)

                    val created = database.createUser(
                        name,
                        user,
                        pass
                    )

                    database.close()

                    if (created) {
                        val intent = android.content.Intent(
                            this,
                            FaceEnrollmentActivity::class.java
                        )

                        intent.putExtra("username", user)

                        startActivity(intent)
                        finish()
                    }
                    else {
                        showMessage("Ese nombre de usuario ya está registrado")
                    }
                }
            }
        }

        btnBack.setOnClickListener {
            finish()
        }
    }

    private fun showMessage(message: String) {
        Toast.makeText(this, message, Toast.LENGTH_LONG).show()
    }
}
