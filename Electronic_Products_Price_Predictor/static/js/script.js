console.log("JS Loaded Successfully");

document.addEventListener("DOMContentLoaded", function () {

    console.log("DOM Fully Loaded");

    // Load dropdown data
    fetch("/api/dropdowns/")
        .then(response => {
            if (!response.ok) {
                throw new Error("Dropdown API not working");
            }
            return response.json();
        })
        .then(data => {

            console.log("Dropdown Data:", data);

            if (data.brands) fillSelect("brand", data.brands);
            if (data.categories) fillSelect("category", data.categories);
            if (data.primaryCategories) fillSelect("primary", data.primaryCategories);
            if (data.availability) fillSelect("availability", data.availability);
            if (data.condition) fillSelect("condition", data.condition);
            if (data.isSale) fillSelect("isSale", data.isSale); // Safe check

        })
        .catch(error => {
            console.error("Dropdown Error:", error);
        });

});


function fillSelect(id, items) {

    let select = document.getElementById(id);

    if (!select) {
        console.warn("Select element not found:", id);
        return;
    }

    if (!items || !Array.isArray(items)) {
        console.warn("Invalid dropdown data for:", id);
        return;
    }

    select.innerHTML = "<option value=''>Select</option>";

    items.forEach(item => {
        let option = document.createElement("option");
        option.value = item;
        option.textContent = item;
        select.appendChild(option);
    });
}


// ================================
// Prediction Function
// ================================

function predictPrice() {

    console.log("Predict button clicked");

    let brand = document.getElementById("brand")?.value;
    let category = document.getElementById("category")?.value;
    let primary = document.getElementById("primary")?.value;
    let availability = document.getElementById("availability")?.value;
    let condition = document.getElementById("condition")?.value;
    let isSale = document.getElementById("isSale")?.value;
    let weight = document.getElementById("weight")?.value;

    // Basic validation
    if (!brand || !category || !primary || !availability || !condition || !weight) {
        alert("Please fill all required fields.");
        return;
    }

    fetch("/api/predict/", {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
            "X-CSRFToken": getCSRFToken()
        },
        body: JSON.stringify({
            brand: brand,
            category: category,
            primary: primary,
            availability: availability,
            condition: condition,
            isSale: isSale,   // if backend uses it
            weight: weight
        })
    })
        .then(response => {
            if (!response.ok) {
                throw new Error("Prediction API Error");
            }
            return response.json();
        })
        .then(data => {

            console.log("Prediction Response:", data);

            if (data.price !== undefined) {
                document.getElementById("result").innerHTML =
                    "Predicted Price: ₹ " + data.price;
            } else if (data.error) {
                document.getElementById("result").innerHTML =
                    "Error: " + data.error;
            } else {
                document.getElementById("result").innerHTML =
                    "Unexpected response from server.";
            }

        })
        .catch(error => {
            console.error("Prediction Error:", error);
            document.getElementById("result").innerHTML =
                "Error predicting price.";
        });

}


// ================================
// CSRF Token Function (Django)
// ================================

function getCSRFToken() {

    let cookieValue = null;

    if (document.cookie && document.cookie !== "") {

        let cookies = document.cookie.split(";");

        for (let i = 0; i < cookies.length; i++) {

            let cookie = cookies[i].trim();

            if (cookie.startsWith("csrftoken=")) {
                cookieValue = cookie.substring("csrftoken=".length);
                break;
            }
        }
    }

    return cookieValue;
}

