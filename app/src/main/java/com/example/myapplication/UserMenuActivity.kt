package com.example.myapplication

import android.content.Intent
import android.os.Bundle
import android.widget.Button
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity

class UserMenuActivity : AppCompatActivity() {

    private lateinit var username: String

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_user_menu)

        username = intent.getStringExtra("username") ?: "Usuario"

        // Bienvenida personalizada
        val txtWelcomeUser =
            findViewById<TextView>(R.id.txtWelcomeUser)

        txtWelcomeUser.text = "¡Bienvenido, $username!"

        // VERIFICAR IDENTIDAD
        val btnVerifyIdentity =
            findViewById<Button>(R.id.btnVerifyIdentity)

        btnVerifyIdentity.setOnClickListener {

            val intent = Intent(
                this,
                MainActivity::class.java
            )

            intent.putExtra("username", username)

            startActivity(intent)
        }

        // ELIMINAR CUENTA
        val btnDeleteAccount =
            findViewById<Button>(R.id.btnDeleteAccount)

        btnDeleteAccount.setOnClickListener {
            showDeleteConfirmation()
        }

        // CERRAR SESIÓN
        val btnLogout =
            findViewById<Button>(R.id.btnLogout)

        btnLogout.setOnClickListener {
            logout()
        }
    }

    // Confirmación de eliminación
    private fun showDeleteConfirmation() {

        AlertDialog.Builder(this)
            .setTitle("Eliminar cuenta")
            .setMessage(
                "¿Estás seguro de que deseas eliminar permanentemente tu cuenta?\n\n" +
                        "Se eliminarán tu usuario, credenciales, plantillas faciales " +
                        "y datos de seguridad asociados.\n\n" +
                        "Esta acción no se puede deshacer."
            )
            .setNegativeButton("Cancelar", null)
            .setPositiveButton("Eliminar cuenta") { _, _ ->
                deleteAccount()
            }
            .show()
    }

    // Eliminar cuenta y todos los datos asociados
    private fun deleteAccount() {

        val database = UserDatabase(this)

        val deleted: Boolean

        try {

            deleted = database.deleteUserAccount(username)

        } catch (e: Exception) {

            android.util.Log.e(
                "FACEVERIFY",
                "Error eliminando cuenta",
                e
            )

            Toast.makeText(
                this,
                "Ocurrió un error al eliminar la cuenta.",
                Toast.LENGTH_LONG
            ).show()

            return

        } finally {

            database.close()
        }

        if (deleted) {

            Toast.makeText(
                this,
                "Cuenta eliminada correctamente",
                Toast.LENGTH_LONG
            ).show()

            // Regresar a bienvenida limpiando la navegación
            val intent = Intent(
                this,
                WelcomeActivity::class.java
            )

            intent.flags =
                Intent.FLAG_ACTIVITY_NEW_TASK or
                        Intent.FLAG_ACTIVITY_CLEAR_TASK

            startActivity(intent)
            finish()

        } else {

            Toast.makeText(
                this,
                "No fue posible eliminar la cuenta. Inténtalo nuevamente.",
                Toast.LENGTH_LONG
            ).show()
        }
    }

    // Cerrar sesión sin eliminar la cuenta
    private fun logout() {

        val intent = Intent(
            this,
            WelcomeActivity::class.java
        )

        intent.flags =
            Intent.FLAG_ACTIVITY_NEW_TASK or
                    Intent.FLAG_ACTIVITY_CLEAR_TASK

        startActivity(intent)
        finish()
    }
}