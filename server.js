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
    fastify.log.debug("processing image");
    const result = await matchImage();
    reply.code(200).send(result);
  });
  mp.on("field", function(key, value) {
    fastify.log.debug("form-data", key, value);
  });
  function handler(field, file, filename, encoding, mimetype) {
    fastify.log.info("File Uploaded", filename);
    pump(file, fs.createWriteStream(QUERY_IMAGE));
  }
});
// FIXME: Redundant route
fastify.get("/image", async (request, reply) => {
  return await matchImage();
});

const matcherList = [];

const matchImage = async () => {
  const queryImage = await canvas.loadImage(QUERY_IMAGE);
  const resultsQuery = await faceapi
    .detectAllFaces(queryImage, faceDetectionOptions)
    .withFaceLandmarks()
    .withFaceDescriptors();
  let matchedFlag = { found: false };
  matcherList.filter(entry => {
    fastify.log.debug(new Date(), "comparing", entry);
    resultsQuery.map(res => {
      const bestMatch = entry.faceMatcher.findBestMatch(res.descriptor);
      if (bestMatch.label !== "unknown") {
        matchedFlag = { found: true, file: entry.file };
      }
    });
  });
  return matchedFlag;
};

// Run the server!
const start = async () => {
  await faceDetectionNet.loadFromDisk(WEIGHTS);
  await faceapi.nets.faceLandmark68Net.loadFromDisk(WEIGHTS);
  await faceapi.nets.faceRecognitionNet.loadFromDisk(WEIGHTS);
  await preloadKnownImage();
  try {
    await fastify.listen(3000);
  } catch (err) {
    fastify.log.error(err);
    process.exit(1);
  }
};
async function preloadKnownImage() {
  const files = await fsPromises.readdir(KNOWN_IMAGE_FOLDER);
  fastify.log.info("preloading images: ", files);
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
      fastify.log.debug("loaded file", f);
      matcherList.push({ faceMatcher, file: f });
    })
  );
}
start();
