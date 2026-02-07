const API = "http://127.0.0.1:8000";

// DOM Elements
const video = document.getElementById("main-video");
const sourceToggle = document.getElementById("source-toggle");
const uploadBtn = document.getElementById("upload-btn-container");
const fileInput = document.getElementById("video-upload");
const qualitySlider = document.getElementById("quality-slider");
const qualityVal = document.getElementById("quality-val");
const lockdownOverlay = document.getElementById("lockdown-overlay");
const recalibrateBtn = document.getElementById("recalibrate-btn");

// Metrics
const riskEl = document.getElementById("riskLevel");
const scoreEl = document.getElementById("driftScore");
const fuelFill = document.getElementById("fuelFill");
const fuelText = document.getElementById("fuelText");
const globalStatus = document.getElementById("global-status");

// --- 1. INITIALIZE THE TREND CHART ---
const ctx = document.getElementById("trendChart").getContext("2d");
const trendChart = new Chart(ctx, {
    type: 'line',
    data: {
        labels: Array(20).fill(''), // Empty labels for clean look
        datasets: [{
            data: Array(20).fill(0),
            borderColor: '#58a6ff',
            borderWidth: 2,
            pointRadius: 0,
            tension: 0.4, // Smooth curves
            fill: true,
            backgroundColor: 'rgba(88, 166, 255, 0.1)'
        }]
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false, // Turn off animation for real-time performance
        plugins: { legend: { display: false } },
        scales: {
            x: { display: false },
            y: { display: false, min: 0, max: 100 }
        }
    }
});

let currentQuality = 1.0;

// --- 2. WEBCAM & CONTROLS ---
function startWebcam() {
    if(navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
        navigator.mediaDevices.getUserMedia({ video: true })
            .then(stream => { video.srcObject = stream; video.play(); })
            .catch(console.error);
    }
}
startWebcam();

if (sourceToggle) {
    sourceToggle.addEventListener("change", (e) => {
        if (e.target.checked) {
            video.srcObject = null;
            uploadBtn.classList.remove("hidden");
        } else {
            uploadBtn.classList.add("hidden");
            startWebcam();
        }
    });
}

if (fileInput) {
    fileInput.addEventListener("change", (e) => {
        const file = e.target.files[0];
        if (file) {
            video.src = URL.createObjectURL(file);
            video.loop = true;
            video.play();
        }
    });
}

if (qualitySlider) {
    qualitySlider.addEventListener("input", (e) => {
        currentQuality = e.target.value / 100;
        if(qualityVal) qualityVal.innerText = Math.round(currentQuality * 100) + "%";
        const blur = (1 - currentQuality) * 15;
        const gray = (1 - currentQuality) * 100;
        video.style.filter = `blur(${blur}px) grayscale(${gray}%) brightness(${50 + (currentQuality * 50)}%)`;
    });
}

// Supervisor Recalibrate
if (recalibrateBtn) {
    recalibrateBtn.addEventListener("click", async () => {
        try {
            await fetch(`${API}/calibrate`, { method: "POST" });
            alert("✅ System Re-Baselined!");
        } catch (e) { alert("Backend Offline"); }
    });
}

// --- 3. MAIN LOOP ---
// 4. MAIN LOOP (Optimized)
setInterval(async () => {
    // Send Frame
    if (video.readyState === 4) {
        // Create a small canvas for processing (320x240 is enough for drift detection)
        const canvas = document.createElement("canvas");
        canvas.width = 320; 
        canvas.height = 240;
        const ctx = canvas.getContext("2d");
        
        // Draw the video frame scaled down
        ctx.drawImage(video, 0, 0, 320, 240);
        
        // Send as low-quality JPEG for speed
        canvas.toBlob(async (blob) => {
            const formData = new FormData();
            formData.append("file", blob, "frame.jpg");
            try {
                await fetch(`${API}/process-frame?quality_flag=${currentQuality}`, { method: "POST", body: formData });
            } catch(e) {}
        }, "image/jpeg", 0.7); // 70% quality
    }

    // Fetch Status (Same as before...)
    try {
        const res = await fetch(`${API}/status`);
        const data = await res.json();

        if(scoreEl) scoreEl.innerText = data.global_drift_score.toFixed(1);
        
        if(riskEl) {
            riskEl.innerText = data.risk_level;
            riskEl.style.color = data.risk_level === "CRITICAL" ? "#cf222e" : "#2da44e";
        }
        
        if(fuelFill) {
            fuelFill.style.width = data.risk_budget + "%";
            if(fuelText) fuelText.innerText = Math.round(data.risk_budget) + "% Remaining";
            fuelFill.style.background = data.risk_level === "CRITICAL" ? "#cf222e" : "#2da44e";
        }

        // Chart Update
        const chartData = trendChart.data.datasets[0].data;
        chartData.shift();
        chartData.push(data.global_drift_score);
        
        if(data.global_drift_score > 60) {
            trendChart.data.datasets[0].borderColor = "#cf222e";
            trendChart.data.datasets[0].backgroundColor = "rgba(207, 34, 46, 0.2)";
        } else {
            trendChart.data.datasets[0].borderColor = "#58a6ff";
            trendChart.data.datasets[0].backgroundColor = "rgba(88, 166, 255, 0.1)";
        }
        trendChart.update();

        // Lockdown
        if(lockdownOverlay) {
            if (data.risk_level === "CRITICAL") lockdownOverlay.classList.remove("hidden");
            else lockdownOverlay.classList.add("hidden");
        }
        
        if(globalStatus) {
            globalStatus.innerText = data.risk_level === "CRITICAL" ? "CRITICAL FAILURE" : "SYSTEM ACTIVE";
            globalStatus.className = data.risk_level === "CRITICAL" ? "status crit" : "status ok";
        }

    } catch(e) {}
}, 1000);
