/** Camera and video-file sources for the <video> element. */

function stopSource(video) {
  video.srcObject?.getTracks().forEach((track) => track.stop());
  video.srcObject = null;
  if (video.src) URL.revokeObjectURL(video.src);
  video.removeAttribute('src');
}

/**
 * @param {'user'|'environment'} facing front ("user") or back camera
 * @returns {Promise<boolean>} whether the picture should be mirrored (front camera)
 */
export async function startCamera(video, facing = 'user') {
  stopSource(video);
  video.srcObject = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: facing, width: { ideal: 1280 }, height: { ideal: 720 } },
    audio: false,
  });
  await video.play();
  return facing === 'user';
}

export async function startVideoFile(video, file) {
  stopSource(video);
  video.src = URL.createObjectURL(file);
  video.loop = true;
  await video.play();
}
