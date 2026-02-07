const API = "http://127.0.0.1:8000";

// IMPACT CHART (Bar)
const impactCtx = document.getElementById("impactChart").getContext("2d");
const impactChart = new Chart(impactCtx, {
    type: 'bar',
    data: {
        labels: ['Visual Blur', 'Occlusion', 'Lighting', 'Sensor Noise'],
        datasets: [{
            label: 'Impact Score',
            data: [0.1, 0.2, 0.1, 0.05],
            backgroundColor: ['#cf222e', '#d29922', '#2da44e', '#58a6ff']
        }]
    },
    options: { responsive: true, maintainAspectRatio: false }
});

// CONFIDENCE CHART (Doughnut)
const confCtx = document.getElementById("confChart").getContext("2d");
const confChart = new Chart(confCtx, {
    type: 'doughnut',
    data: {
        labels: ['High Confidence', 'Low Confidence'],
        datasets: [{
            data: [90, 10],
            backgroundColor: ['#2da44e', '#cf222e'],
            borderWidth: 0
        }]
    },
    options: { responsive: true, maintainAspectRatio: false }
});

setInterval(async () => {
    try {
        const sRes = await fetch(`${API}/status`);
        const status = await sRes.json();
        
        // Mocking dynamic data for demo
        const drift = status.global_drift_score;
        
        // Update Impact Bar
        impactChart.data.datasets[0].data = [
            Math.min(100, drift * 1.2), // Blur rises with drift
            Math.random() * 20, 
            Math.random() * 10, 
            Math.random() * 5
        ];
        impactChart.update();

        // Update Confidence Doughnut
        // Parse "95% (Real-time)" -> 95
        const confVal = parseInt(status.model_confidence) || 0;
        confChart.data.datasets[0].data = [confVal, 100 - confVal];
        confChart.update();

    } catch (e) {}
}, 2000);