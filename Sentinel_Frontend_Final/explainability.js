const API = "http://127.0.0.1:8000";

const topFeature = document.getElementById("topFeature"); // Check HTML ID
const operatorMsg = document.getElementById("operatorMsg"); // Check HTML ID
const list = document.getElementById("featureList"); // Check HTML ID

async function loadExplainability() {
  try {
    const res = await fetch(`${API}/explainability`);
    const data = await res.json();

    if (topFeature) topFeature.innerText = data.top_driving_feature;
    if (operatorMsg) operatorMsg.innerText = data.operator_message;

    if (list) {
        list.innerHTML = "";
        for (const [key, val] of Object.entries(data.all_feature_scores)) {
            const li = document.createElement("li");
            li.style.padding = "10px";
            li.style.borderBottom = "1px solid #2a3446";
            li.style.display = "flex";
            li.style.justifyContent = "space-between";
            
            // Visual Bar
            const width = Math.min(100, val * 100);
            const color = val > 0.5 ? "#cf222e" : "#2da44e";
            
            li.innerHTML = `
                <span>${key}</span>
                <div style="display:flex; align-items:center; gap:10px;">
                    <div style="width:100px; height:6px; background:#0c1117; border-radius:3px;">
                        <div style="width:${width}%; height:100%; background:${color};"></div>
                    </div>
                    <span style="color:#8b949e;">${val.toFixed(2)}</span>
                </div>
            `;
            list.appendChild(li);
        }
    }
  } catch (e) { console.log("Explainability offline..."); }
}

setInterval(loadExplainability, 2000);