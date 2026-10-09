const SAMPLE_CASES = {
    0: {
        age: 41,
        sex: 0,
        cp: 2,
        trestbps: 130,
        chol: 204,
        fbs: 0,
        restecg: 2,
        thalach: 172,
        exang: 0,
        oldpeak: 1.4,
        slope: 1,
        ca: 0,
        thal: 3
    },
    1: {
        age: 67,
        sex: 1,
        cp: 4,
        trestbps: 160,
        chol: 286,
        fbs: 0,
        restecg: 2,
        thalach: 108,
        exang: 1,
        oldpeak: 1.5,
        slope: 2,
        ca: 3,
        thal: 3
    }
};

function loadSampleCase(caseType) {
    const data = SAMPLE_CASES[caseType];
    if (!data) return;

    for (const [key, val] of Object.entries(data)) {
        const el = document.getElementById(key);
        if (el) {
            el.value = val;
        }
    }

    submitDiagnosis();
}

async function submitDiagnosis() {
    const btnSubmit = document.getElementById("btn-submit");
    const btnText = document.getElementById("btn-text");
    const btnSpinner = document.getElementById("btn-spinner");

    const fields = [
        "age", "sex", "cp", "trestbps", "chol", "fbs",
        "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"
    ];

    const payload = {};
    for (const f of fields) {
        const el = document.getElementById(f);
        if (!el || el.value === "") {
            alert(`Parameter ${f} wajib diisi.`);
            return;
        }
        payload[f] = parseFloat(el.value);
    }

    btnSubmit.disabled = true;
    btnText.innerText = "Mengevaluasi Model Neural Network...";
    btnSpinner.classList.remove("hidden");

    try {
        const response = await fetch("/api/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(payload)
        });

        const result = await response.json();
        if (!response.ok || !result.success) {
            throw new Error(result.error || "Gagal memproses prediksi");
        }

        renderDiagnosisResult(result);

    } catch (err) {
        alert("Terjadi kesalahan: " + err.message);
    } finally {
        btnSubmit.disabled = false;
        btnText.innerText = "Klasifikasikan Risiko Pasien";
        btnSpinner.classList.add("hidden");
    }
}

function renderDiagnosisResult(data) {
    const placeholder = document.getElementById("result-placeholder");
    const content = document.getElementById("result-content");

    placeholder.classList.add("hidden");
    content.classList.remove("hidden");

    const now = new Date();
    document.getElementById("res-timestamp").innerText = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

    const statusCard = document.getElementById("status-card");
    const title = document.getElementById("res-title");
    const kesimpulan = document.getElementById("res-kesimpulan");
    const modelBadge = document.getElementById("res-model-badge");

    modelBadge.innerText = data.model_digunakan.toUpperCase();
    title.innerText = data.analisis.status_text.toUpperCase();
    kesimpulan.innerText = data.analisis.kesimpulan;

    statusCard.className = "p-5 rounded-xl border text-center space-y-2 transition-all " + data.analisis.status_badge;

    const pSakit = data.probabilitas.sakit;
    const pSehat = data.probabilitas.sehat;

    document.getElementById("prob-sakit-txt").innerText = `${pSakit}%`;
    document.getElementById("prob-sakit-bar").style.width = `${pSakit}%`;

    document.getElementById("prob-sehat-txt").innerText = `${pSehat}%`;
    document.getElementById("prob-sehat-bar").style.width = `${pSehat}%`;

    const recList = document.getElementById("res-rekomendasi");
    recList.innerHTML = "";
    (data.analisis.rekomendasi || []).forEach(item => {
        const li = document.createElement("li");
        li.className = "flex items-start gap-2";
        li.innerHTML = `<span class="text-rose-800 font-bold">•</span><span>${item}</span>`;
        recList.appendChild(li);
    });

    if (window.lucide) {
        lucide.createIcons();
    }
}
