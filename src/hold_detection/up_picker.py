"""Colab canvas for selecting a vertical line on a registered image."""

import base64
from io import BytesIO
import json

from PIL import Image


def picker_frame(path):
    with Image.open(path) as source:
        width, height = source.size
        preview = source.convert("RGB")
    preview.thumbnail((1050, 1050), Image.Resampling.LANCZOS)
    with BytesIO() as buffer:
        preview.save(buffer, "JPEG", quality=88)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return {"url": "data:image/jpeg;base64," + encoded, "width": width, "height": height}


def picker_javascript(frames, callback_prefix):
    arguments = json.dumps([frames, callback_prefix]).replace("<", "\\u003c")
    return r"""(async ([frames, prefix]) => {
  const root = document.createElement('section');
  root.innerHTML = `
    <p><label>Registered frame <select></select></label></p>
    <p>Click the bottom, then the top of a real vertical line.</p>
    <canvas style="display:block;max-width:100%;height:auto;cursor:crosshair"></canvas>
    <p><button data-action="apply" disabled>Apply up direction</button>
    <button data-action="reset">Clear points</button>
    <button data-action="auto">Use automatic estimate</button></p>
    <p role="status"></p>`;
  (document.querySelector('#output-area') || document.body).append(root);
  const select = root.querySelector('select');
  const canvas = root.querySelector('canvas');
  const context = canvas.getContext('2d');
  const apply = root.querySelector('[data-action=apply]');
  const auto = root.querySelector('[data-action=auto]');
  const status = root.querySelector('[role=status]');
  const image = new Image();
  const request = async (action, args) => {
    const result = await google.colab.kernel.invokeFunction(prefix + action, args, {});
    const data = result?.data?.['application/json'];
    if (!data) throw new Error('The notebook did not return a calibration response.');
    return data;
  };
  let frame = null;
  let points = [];
  let busy = false;

  const draw = () => {
    if (!frame) return;
    context.drawImage(image, 0, 0, canvas.width, canvas.height);
    const projected = points.map(([x, y]) => [x * canvas.width / frame.width,
                                             y * canvas.height / frame.height]);
    context.strokeStyle = '#fbbf24';
    context.fillStyle = '#fbbf24';
    context.lineWidth = 4;
    context.font = 'bold 18px sans-serif';
    if (projected.length === 2) {
      const [[x1, y1], [x2, y2]] = projected;
      const angle = Math.atan2(y2 - y1, x2 - x1);
      context.beginPath();
      context.moveTo(x1, y1);
      context.lineTo(x2, y2);
      for (const offset of [-0.55, 0.55]) {
        context.moveTo(x2, y2);
        context.lineTo(x2 - 20 * Math.cos(angle + offset), y2 - 20 * Math.sin(angle + offset));
      }
      context.stroke();
    }
    projected.forEach(([x, y], index) => {
      context.beginPath();
      context.arc(x, y, 5, 0, 2 * Math.PI);
      context.fill();
      context.fillText(index === 0 ? '1 Bottom' : '2 Top', x + 8, Math.max(20, y - 8));
    });
    apply.disabled = busy || points.length !== 2;
  };
  const action = async (message, callback) => {
    busy = select.disabled = auto.disabled = apply.disabled = true;
    status.textContent = message;
    try { await callback(); }
    catch (error) { status.textContent = error.message || String(error); }
    finally {
      busy = select.disabled = auto.disabled = false;
      draw();
    }
  };
  const load = () => action('Loading registered photo…', async () => {
    points = [];
    frame = null;
    const payload = await request('frame', [select.value]);
    image.src = payload.url;
    await image.decode();
    canvas.width = image.naturalWidth;
    canvas.height = image.naturalHeight;
    frame = payload;
    status.textContent = 'Click point 1 at the bottom.';
  });
  frames.forEach(name => select.add(new Option(name, name)));
  select.onchange = load;
  canvas.onclick = event => {
    if (busy || !frame || points.length === 2) return;
    const box = canvas.getBoundingClientRect();
    points.push([(event.clientX - box.left) * frame.width / box.width,
                 (event.clientY - box.top) * frame.height / box.height]);
    status.textContent = points.length === 1 ? 'Click point 2 at the top.' : 'Check the arrow and apply.';
    draw();
  };
  root.querySelector('[data-action=reset]').onclick = () => {
    if (busy) return;
    points = [];
    status.textContent = 'Click point 1 at the bottom.';
    draw();
  };
  const save = reference => action('Updating scene.json…', async () => {
    const result = await request('apply', [reference]);
    status.textContent = result.message;
  });
  apply.onclick = () => save({image: select.value, bottom_px: points[0], top_px: points[1]});
  auto.onclick = () => save(null);
  await load();
})""" + "(" + arguments + ");"
