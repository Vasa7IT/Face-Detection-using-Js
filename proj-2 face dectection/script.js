// Get the video element from the HTML document
const video = document.getElementById("video");

// Load the required models asynchronously
Promise.all([
  faceapi.nets.ssdMobilenetv1.loadFromUri("/models"),
  faceapi.nets.faceRecognitionNet.loadFromUri("/models"),
  faceapi.nets.faceLandmark68Net.loadFromUri("/models"),
]).then(startWebcam);

// Function to start the webcam and stream video
function startWebcam() {
  navigator.mediaDevices
    .getUserMedia({
      video: true,
      audio: false,
    })
    .then((stream) => {
      video.srcObject = stream;
    })
    .catch((error) => {
      console.error(error);
    });
}

// Function to get labeled face descriptions
function getLabeledFaceDescriptions() {
  const labels = ["kamal", "dhoni", "suriya"];
  return Promise.all(
    labels.map(async (label) => {
      const descriptions = [];
      for (let i = 1; i <= 2; i++) {
        // Fetch face images for each label
        const img = await faceapi.fetchImage(`./labels/${label}/${i}.png`);
        // Detect facial landmarks and descriptors for each face image
        const detections = await faceapi
          .detectSingleFace(img)
          .withFaceLandmarks()
          .withFaceDescriptor();
        // Store the face descriptors
        descriptions.push(detections.descriptor);
      }
      // Create LabeledFaceDescriptors object for each label
      return new faceapi.LabeledFaceDescriptors(label, descriptions);
    })
  );
}

// Event listener for the "play" event of the video element
video.addEventListener("play", async () => {
  // Get labeled face descriptors
  const labeledFaceDescriptors = await getLabeledFaceDescriptions();
  // Create a FaceMatcher object with labeled face descriptors
  const faceMatcher = new faceapi.FaceMatcher(labeledFaceDescriptors);
  // Create a canvas element to draw face recognition results
  const canvas = faceapi.createCanvasFromMedia(video);
  document.body.append(canvas);
  // Set display size for the canvas
  const displaySize = { width: video.width, height: video.height };
  faceapi.matchDimensions(canvas, displaySize);
  // Perform face detection and recognition in intervals

  setInterval(async () => {
    const detections = await faceapi
      .detectAllFaces(video)
      .withFaceLandmarks()
      .withFaceDescriptors();
    // Resize detected faces to match the display size
    const resizedDetections = faceapi.resizeResults(detections, displaySize);
    // Clear the canvas before drawing new results
    canvas.getContext("2d").clearRect(0, 0, canvas.width, canvas.height);
    // Match detected faces with labeled face descriptors
    const results = resizedDetections.map((d) => {
      return faceMatcher.findBestMatch(d.descriptor);
    });
    // Draw boxes around detected faces with labels
    results.forEach((result, i) => {
      const box = resizedDetections[i].detection.box;
      const drawBox = new faceapi.draw.DrawBox(box, {
        label: result,
      });
      drawBox.draw(canvas);
    });
  }, 100); // Set interval duration to 100 milliseconds
});
