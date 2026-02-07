// supervisor.js - Dynamic Governance Visuals
const API = "http://127.0.0.1:8000";

// 1. RELIABILITY CURVE (Survival Analysis)
const relCtx = document.getElementById("reliabilityChart").getContext("2d");
const relChart = new Chart(relCtx, {
    type: 'line',
    data: {
        labels: ['0h', '1h', '2h', '3h', '4h', '5h'],
        datasets: [{
            label: 'Reliability P(t)',
            data: [1.0, 0.98, 0.95, 0.91, 0.85, 0.78], 
            borderColor: '#2da44e',
            backgroundColor: 'rgba(45, 164, 78, 0.1)',
            fill: true,
            tension: 0.4
        }]
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false, // Important for live updates
        plugins: { legend: { display: false } },
        scales: { y: { min: 0, max: 1 } }
    }
});

// 2. DISTRIBUTION SHIFT (KS-Test)
const distCtx = document.getElementById("distChart").getContext("2d");
const distChart = new Chart(distCtx, {
    type: 'line',
    data: {
        labels: Array.from({length: 50}, (_, i) => i),
        datasets: [
            {
                label: 'Baseline (Training)',
                data: Array.from({length: 50}, (_, i) => Math.exp(-Math.pow(i-20, 2)/50)), 
                borderColor: '#58a6ff',
                borderWidth: 2,
                pointRadius: 0,
                fill: true,
                backgroundColor: 'rgba(88, 166, 255, 0.1)'
            },
            {
                label: 'Live Input',
                data: Array.from({length: 50}, (_, i) => Math.exp(-Math.pow(i-20, 2)/50)), // Starts same as baseline
                borderColor: '#d29922',
                borderWidth: 2,
                pointRadius: 0,
                borderDash: [5, 5],
                fill: false
            }
        ]
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        scales: { y: { display: false }, x: { display: false } }
    }
});

// 3. LIVE UPDATE LOOP
setInterval(async () => {
    try {
        const res = await fetch(`${API}/status`);
        const data = await res.json();
        const drift = data.global_drift_score;

        // --- UPDATE RELIABILITY CURVE ---
        // Logic: As drift increases, reliability drops faster
        const decayFactor = 1.0 - (drift / 200); // 0.0 drift = 1.0 factor, 100 drift = 0.5 factor
        
        relChart.data.datasets[0].data = [
            1.0, 
            0.98 * decayFactor, 
            0.95 * Math.pow(decayFactor, 2), 
            0.91 * Math.pow(decayFactor, 3), 
            0.85 * Math.pow(decayFactor, 4), 
            0.78 * Math.pow(decayFactor, 5)
        ];
        
        // Color change if critical
        relChart.data.datasets[0].borderColor = drift > 60 ? '#cf222e' : '#2da44e';
        relChart.data.datasets[0].backgroundColor = drift > 60 ? 'rgba(207, 34, 46, 0.1)' : 'rgba(45, 164, 78, 0.1)';
        relChart.update();

        // --- UPDATE DISTRIBUTION SHIFT ---
        // Logic: Shift the "Live Input" curve to the right as drift increases
        const shift = drift / 5; // Max shift of 20 units
        distChart.data.datasets[1].data = Array.from({length: 50}, (_, i) => 
            Math.exp(-Math.pow(i - (20 + shift), 2) / 50) * (1 - drift/200) // Also flatten height
        );
        distChart.update();

    } catch (e) {}
}, 1000);