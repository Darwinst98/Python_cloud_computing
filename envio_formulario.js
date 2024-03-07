document.getElementById("diabetesForm").addEventListener("submit", function(event) {
    event.preventDefault(); // Evitar el envío del formulario

    // Obtener los datos del formulario
    const formData = new FormData(event.target);
    const jsonData = {};

    // Iterar sobre los datos del formulario
    formData.forEach((value, key) => {
        // Reemplazar comas por puntos en los valores numéricos
        const cleanedValue = typeof value === 'float' && value.includes(',') ? value.replace(',', '.') : value;
        jsonData[key] = cleanedValue;
    });

    // Enviar los datos al servidor en formato JSON
    fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify(jsonData)
    })
    .then(response => response.json())  // Convertir la respuesta a JSON
    .then(data => {
        // Mostrar la respuesta del servidor
        document.getElementById("result") = "Tiene diabetes";
    })
    .catch(error => console.error("Error:", error));  // Manejar errores
});
