const fileInput = document.getElementById("file");
const drop = document.getElementById("drop");
const preview = document.getElementById("preview");
const dropText = document.getElementById("dropText");
const form = document.getElementById("controls");
const go = document.getElementById("go");
const statusText = document.getElementById("status");
const results = document.getElementById("results");

let selectedFile = null;

function showPreview(file) {
  if (!file || !file.type.startsWith("image/")) {
    statusText.textContent = "Please select a valid image.";
    return;
  }

  if (file.size > 10 * 1024 * 1024) {
    statusText.textContent = "Image must be smaller than 10 MB.";
    return;
  }

  selectedFile = file;
  preview.src = URL.createObjectURL(file);
  preview.hidden = false;
  dropText.hidden = true;
  go.disabled = false;
  statusText.textContent = "Image selected successfully.";
  results.hidden = true;
}

fileInput.addEventListener("change", () => {
  showPreview(fileInput.files[0]);
});

["dragover", "dragenter"].forEach(event => {
  drop.addEventListener(event, e => {
    e.preventDefault();
    drop.style.borderColor = "#20d6a0";
  });
});

["dragleave", "drop"].forEach(event => {
  drop.addEventListener(event, e => {
    e.preventDefault();
    drop.style.borderColor = "";
  });
});

drop.addEventListener("drop", e => {
  const file = e.dataTransfer.files[0];
  showPreview(file);
});

const sigma = document.getElementById("sigma");
const sigmaVal = document.getElementById("sigmaVal");
const clip = document.getElementById("clip");
const clipVal = document.getElementById("clipVal");

sigma.addEventListener("input", () => {
  sigmaVal.textContent = sigma.value;
});

clip.addEventListener("input", () => {
  clipVal.textContent = clip.value;
});

form.addEventListener("submit", async e => {
  e.preventDefault();

  if (!selectedFile) {
    statusText.textContent = "Please upload an image first.";
    return;
  }

  const data = new FormData();
  data.append("image", selectedFile);
  data.append(
    "method",
    form.querySelector('input[name="method"]:checked').value
  );
  data.append("sigma", sigma.value);
  data.append("clip", clip.value);

  go.disabled = true;
  statusText.textContent = "Enhancing image... Please wait.";

  try {
    const response = await fetch("/enhance", {
      method: "POST",
      body: data
    });

    const result = await response.json();

    if (!response.ok) {
      throw new Error(result.error || "Image enhancement failed.");
    }

    document.getElementById("imgBefore").src =
      result.original || preview.src;

    document.getElementById("imgAfter").src =
      result.enhanced;

    document.getElementById("imgI").src =
      result.input;

    document.getElementById("imgL").src =
      result.illumination;

    document.getElementById("imgR").src =
      result.reflectance;

    const metrics = document.getElementById("metrics");
    metrics.innerHTML = "";

    const names = [
      ["Brightness", "brightness"],
      ["Contrast", "contrast"],
      ["Entropy", "entropy"]
    ];

    names.forEach(([label, key]) => {
      const before = result.metrics.original[key];
      const after = result.metrics.enhanced[key];

      metrics.innerHTML += `
        <tr>
          <td>${label}</td>
          <td>${Number(before).toFixed(2)}</td>
          <td>${Number(after).toFixed(2)}</td>
        </tr>`;
    });

    const download = document.getElementById("download");
    download.href = result.enhanced;

    results.hidden = false;
    statusText.textContent = "Image enhanced successfully.";

    updateCompare();

    results.scrollIntoView({
      behavior: "smooth",
      block: "start"
    });

  } catch (error) {
    statusText.textContent = error.message;
  } finally {
    go.disabled = false;
  }
});

const cmpRange = document.getElementById("cmpRange");
const beforeClip = document.getElementById("clip-before");
const handle = document.getElementById("handle");

function updateCompare() {
  const value = cmpRange.value;
  beforeClip.style.width = value + "%";
  handle.style.left = value + "%";
}

cmpRange.addEventListener("input", updateCompare);
updateCompare();
