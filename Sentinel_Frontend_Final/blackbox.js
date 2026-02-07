const API = "http://127.0.0.1:8000";
const logBody = document.getElementById("logBody");

async function loadLogs() {
  try {
    const res = await fetch(`${API}/logs`);
    const data = await res.json();

    if (logBody) {
        logBody.innerHTML = "";
        data.logs.forEach(log => {
          const tr = document.createElement("tr");
          // Style the Severity Badge
          let badgeClass = "ok";
          if(log.severity === "CRITICAL") badgeClass = "crit";
          if(log.severity === "WARNING") badgeClass = "warn";
          
          tr.innerHTML = `
            <td>${log.timestamp}</td>
            <td><span class="status ${badgeClass}" style="font-size:0.8rem;">${log.severity}</span></td>
            <td>${log.action_taken}</td>
            <td>${log.root_cause}</td>
          `;
          logBody.appendChild(tr);
        });
    }
  } catch {}
}

setInterval(loadLogs, 3000);