const faceapi = require("face-api.js");
const pump = require("pump");
const fs = require("fs");
const fsPromises = fs.promises;

const { canvas, faceDetectionNet, faceDetectionOptions } = require("./commons");
const KNOWN_IMAGE_FOLDER = "./images/known";
const QUERY_IMAGE = "./images/temp/uploaded.jpg";
const WEIGHTS = "./weights";
const fastify = require("fastify")({ logger: true });
fastify.register(require("fastify-multipart"));

fastify.post("/image", async (req, reply) => {
  const mp = req.multipart(handler, async err => {
    console.log("upload completed");
    const result = await matchImage();
    reply.code(200).send(result);
  });
  mp.on("field", function(key, value) {
    console.log("form-data", key, value);
  });
  function handler(field, file, filename, encoding, mimetype) {
    console.log("File Uploaded");
    pump(file, fs.createWriteStream(QUERY_IMAGE));
  }
});
// Declare a route
fastify.get("/image", async (request, reply) => {
  return await matchImage();
});

const matchImage = async () => {
  const queryImage = await canvas.loadImage(QUERY_IMAGE);
  const resultsQuery = await faceapi
    .detectAllFaces(queryImage, faceDetectionOptions)
    .withFaceLandmarks()
    .withFaceDescriptors();
  let matchedFlag = { found: false };
  const files = await fsPromises.readdir(KNOWN_IMAGE_FOLDER);
  console.log("processing: ", files);
  await Promise.all(
    files.map(async f => {
      const referenceImage = await canvas.loadImage(
        `${KNOWN_IMAGE_FOLDER}/${f}`
      );
      const resultsRef = await faceapi
        .detectAllFaces(referenceImage, faceDetectionOptions)
        .withFaceLandmarks()
        .withFaceDescriptors();
      const faceMatcher = new faceapi.FaceMatcher(resultsRef);
      resultsQuery.map(res => {
        const bestMatch = faceMatcher.findBestMatch(res.descriptor);
        console.log(bestMatch);
        if (bestMatch.label !== "unknown") {
          matchedFlag = { found: true, file: f };
        }
      });
    })
  );
  return matchedFlag;
};

// Run the server!
const start = async () => {
  await faceDetectionNet.loadFromDisk(WEIGHTS);
  await faceapi.nets.faceLandmark68Net.loadFromDisk(WEIGHTS);
  await faceapi.nets.faceRecognitionNet.loadFromDisk(WEIGHTS);
  try {
    await fastify.listen(3000);
    fastify.log.info(`server listening on ${fastify.server.address().port}`);
  } catch (err) {
    fastify.log.error(err);
    process.exit(1);
  }
};
start();
