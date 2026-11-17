const fileInput = document.getElementById("file");
const drop = document.getElementById("drop");
const preview = document.getElementById("preview");
const dropText = document.getElementById("dropText");
const form = document.getElementById("controls");
const go = document.getElementById("go");
const statusText = document.getElementById("status");
const results = document.getElementById("results");

let selectedFile = null;
let previewURL = null;

function selectImage(file) {
  if (!file || !file.type.startsWith("image/")) {
    statusText.textContent = "Please select a valid image.";
    return;
  }

  if (file.size > 10 * 1024 * 1024) {
    statusText.textContent = "Maximum image size is 10 MB.";
    return;
  }

  selectedFile = file;

  if (previewURL) URL.revokeObjectURL(previewURL);
  previewURL = URL.createObjectURL(file);

  preview.src = previewURL;
  preview.hidden = false;
  dropText.hidden = true;
  go.disabled = false;
  results.hidden = true;

  statusText.textContent = "Image selected successfully.";
}

fileInput.addEventListener("change", () => {
  selectImage(fileInput.files[0]);
});

drop.addEventListener("dragover", event => {
  event.preventDefault();
  drop.style.borderColor = "#22d6a0";
});

drop.addEventListener("dragleave", () => {
  drop.style.borderColor = "";
});

drop.addEventListener("drop", event => {
  event.preventDefault();
  drop.style.borderColor = "";
  selectImage(event.dataTransfer.files[0]);
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

form.querySelectorAll('input[name="method"]').forEach(input => {
  input.addEventListener("change", () => {
    sigma.disabled = input.value === "msr" && input.checked;
  });
});

sigma.disabled = true;

function showMetric(label, before, after) {
  const row = document.createElement("tr");

  [label, before, after].forEach(value => {
    const cell = document.createElement("td");
    cell.textContent = value;
    row.appendChild(cell);
  });

  document.getElementById("metrics").appendChild(row);
}

form.addEventListener("submit", async event => {
  event.preventDefault();

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
  statusText.textContent = "Processing image... Please wait.";

  try {
    const response = await fetch("/enhance", {
      method: "POST",
      body: data
    });

    const result = await response.json();

    if (!response.ok) {
      throw new Error(result.error || "Enhancement failed.");
    }

    document.getElementById("imgBefore").src = result.original;
    document.getElementById("imgAfter").src = result.enhanced;
    document.getElementById("imgI").src = result.input;
    document.getElementById("imgL").src = result.illumination;
    document.getElementById("imgR").src = result.reflectance;

    const metrics = document.getElementById("metrics");
    metrics.replaceChildren();

    showMetric(
      "Brightness",
      result.metrics.original.brightness,
      result.metrics.enhanced.brightness
    );

    showMetric(
      "Contrast",
      result.metrics.original.contrast,
      result.metrics.enhanced.contrast
    );

    showMetric(
      "Entropy",
      result.metrics.original.entropy,
      result.metrics.enhanced.entropy
    );

    document.getElementById("download").href = result.enhanced;

    document.getElementById("time").textContent =
      "Processing time: " + result.processing_time + " seconds";

    results.hidden = false;
    updateCompare();

    statusText.textContent = "Enhancement completed successfully.";

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
  const value = Number(cmpRange.value);
  const comparison = document.getElementById("cmp");
  const width = comparison.clientWidth;

  beforeClip.style.width = value + "%";
  handle.style.left = value + "%";

  document.getElementById("imgBefore").style.width = width + "px";
}

cmpRange.addEventListener("input", updateCompare);

window.addEventListener("resize", updateCompare);
