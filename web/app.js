const $ = (id) => document.getElementById(id);
const state = { image: null, markers: [], result: null, drag: null, runId: null };
const fields = ['summary', 'limitations', 'advice1', 'advice2', 'imageSource', 'imageLicense', 'sceneNote', 'toolSource', 'referenceSource'];

function setMessage(message, error = false) {
  $('apiMessage').textContent = message;
  $('apiMessage').classList.toggle('error', error);
}

function loadImage(file) {
  if (!file) return;
  if (!['image/png', 'image/jpeg', 'image/webp'].includes(file.type) || file.size > 10 * 1024 * 1024) {
    setMessage('请选择 10 MB 以内的 PNG、JPEG 或 WebP 图片。', true);
    return;
  }
  const reader = new FileReader();
  reader.onload = () => {
    state.image = { name: file.name, mime: file.type, dataUrl: reader.result };
    $('imageInput').value = '';
    state.markers = [];
    state.result = null;
    state.runId = null;
    $('runInfo').textContent = '';
    $('runInfo').classList.add('hidden');
    $('reviewConfirmed').checked = false;
    $('muralImage').src = reader.result;
    $('sheetImage').src = reader.result;
    $('emptyImage').classList.add('hidden');
    $('imageStage').classList.remove('hidden');
    $('dropZone').classList.add('loaded');
    $('analyzeBtn').disabled = false;
    $('modelState').textContent = '待分析';
    fields.forEach(id => { $(id).value = ''; });
    $('modelFindings').replaceChildren(); $('modelFindings').classList.add('hidden');
    renderMarkers();
    setMessage('图片已载入。可先手动标记区域，也可以调用模型获取观察建议。');
  };
  reader.readAsDataURL(file);
}

$('dropZone').addEventListener('click', (event) => {
  if (!state.image || event.target === $('emptyImage') || $('emptyImage').contains(event.target)) $('imageInput').click();
});
$('dropZone').addEventListener('dragover', (event) => event.preventDefault());
$('dropZone').addEventListener('drop', (event) => { event.preventDefault(); loadImage(event.dataTransfer.files[0]); });
$('imageInput').addEventListener('change', (event) => loadImage(event.target.files[0]));
$('replaceImage').addEventListener('click', () => $('imageInput').click());

function point(event) {
  const r = $('imageStage').getBoundingClientRect();
  return { x: Math.max(0, Math.min(100, (event.clientX - r.left) / r.width * 100)), y: Math.max(0, Math.min(100, (event.clientY - r.top) / r.height * 100)) };
}

$('markerLayer').addEventListener('pointerdown', (event) => {
  if (!state.image || state.markers.length >= 3 || event.target.closest('.marker')) return;
  const p = point(event);
  state.drag = { start: p, current: p };
  $('markerLayer').setPointerCapture(event.pointerId);
});
$('markerLayer').addEventListener('pointermove', (event) => {
  if (!state.drag) return;
  state.drag.current = point(event);
  renderMarkers();
});
$('markerLayer').addEventListener('pointerup', (event) => {
  if (!state.drag) return;
  const end = point(event), start = state.drag.start;
  let x = Math.min(start.x, end.x), y = Math.min(start.y, end.y);
  let w = Math.abs(start.x - end.x), h = Math.abs(start.y - end.y);
  if (w < 3 && h < 3) { x = Math.max(0, x - 7); y = Math.max(0, y - 7); w = 14; h = 14; }
  state.markers.push({ x, y, w: Math.max(3, w), h: Math.max(3, h), phenomenon: '', evidence: '', status: '疑似发现' });
  state.drag = null;
  renderMarkers();
});

function markerFromRegion(region) {
  const box = region.bbox;
  const [x1, y1, x2, y2] = Array.isArray(box) ? box : [box.x1, box.y1, box.x2, box.y2];
  const pending = region.pending_review || region.confidence === '低';
  return {
    x: x1 * 100, y: y1 * 100, w: (x2 - x1) * 100, h: (y2 - y1) * 100,
    phenomenon: `${region.disease}：${region.observation}`,
    evidence: [region.impact, ...(region.review_notes || [])].filter(Boolean).join('；'),
    gradeLabel: `${region.confidence}置信度${pending ? ' · 人工复核' : ''}`,
    status: pending ? '人工复核' : '自动判定',
  };
}

function markerElement(marker, index, clickable = false) {
  const el = document.createElement('div');
  el.className = 'marker';
  el.style.cssText = `left:${marker.x}%;top:${marker.y}%;width:${marker.w}%;height:${marker.h}%`;
  el.textContent = index + 1;
  if (clickable) el.addEventListener('click', (event) => { event.stopPropagation(); $(`phenomenon-${index}`)?.focus(); });
  return el;
}

function renderMarkers() {
  $('markerLayer').replaceChildren(...state.markers.map((m, i) => markerElement(m, i, true)));
  $('sheetMarkers').replaceChildren(...state.markers.map((m, i) => markerElement(m, i)));
  if (state.drag) {
    const a = state.drag.start, b = state.drag.current;
    $('markerLayer').append(markerElement({ x: Math.min(a.x, b.x), y: Math.min(a.y, b.y), w: Math.abs(a.x-b.x), h: Math.abs(a.y-b.y) }, state.markers.length));
  }
  $('markerCount').textContent = `${state.markers.length} / 3 处`;
  renderFindings();
}

function renderFindings() {
  const list = $('findingList');
  list.replaceChildren();
  if (!state.markers.length) { list.innerHTML = '<div class="empty-list">标记区域后，在这里填写观察现象和依据。</div>'; return; }
  state.markers.forEach((m, i) => {
    const card = document.createElement('div'); card.className = 'finding';
    card.innerHTML = `<div class="finding-title"><strong>区域 ${i + 1}</strong><button type="button" class="text-button remove">移除</button></div><label>观察现象<input id="phenomenon-${i}" placeholder="如：线条中断、局部颜色变浅"></label><label>图像依据<input placeholder="说明能从原图看到什么"></label><label>判断等级<select><option>自动判定</option><option>人工复核</option><option>疑似发现</option><option>明确发现</option><option>无法判断</option></select></label>`;
    if (m.gradeLabel) card.querySelector('strong').textContent = `区域 ${i + 1} · ${m.gradeLabel}`;
    const inputs = card.querySelectorAll('input');
    inputs[0].value = m.phenomenon; inputs[1].value = m.evidence; card.querySelector('select').value = m.status;
    inputs[0].addEventListener('input', e => m.phenomenon = e.target.value);
    inputs[1].addEventListener('input', e => m.evidence = e.target.value);
    card.querySelector('select').addEventListener('change', e => m.status = e.target.value);
    card.querySelector('.remove').addEventListener('click', () => { state.markers.splice(i, 1); renderMarkers(); });
    list.append(card);
  });
}

$('clearMarkers').addEventListener('click', () => { state.markers = []; renderMarkers(); });

$('analyzeBtn').addEventListener('click', async () => {
  if (!state.image) return;
  if (!$('imageSource').value.trim() || !$('imageLicense').value.trim()) {
    setMessage('请先填写图片来源和授权范围。', true);
    return;
  }
  const analyzedImage = state.image;
  $('analyzeBtn').disabled = true; $('modelState').textContent = '分析中';
  setMessage('正在调用模型，请稍候。');
  try {
    const response = await fetch('/api/analyze', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ mime: state.image.mime, image: state.image.dataUrl.split(',')[1], image_source: $('imageSource').value, image_license: $('imageLicense').value, scene_note: $('sceneNote').value }) });
    const data = await response.json();
    if (state.image !== analyzedImage) return;
    if (!response.ok) throw new Error(data.error || '模型调用失败');
    state.runId = data.run_id || null;
    $('runInfo').textContent = state.runId ? `运行编号：${state.runId}` : '';
    $('runInfo').classList.toggle('hidden', !state.runId);
    $('reviewConfirmed').checked = false;
    if (!data.card) {
      if (state.result) {
        state.markers = [];
        for (const id of ['summary', 'limitations', 'advice1', 'advice2', 'toolSource']) $(id).value = '';
      }
      state.result = null;
      $('modelFindings').replaceChildren();
      $('modelFindings').classList.add('hidden');
      $('modelState').textContent = '待人工检查';
      setMessage(data.message || '分析完成，暂无可确认的关注区域。');
      syncSheet();
      return;
    }
    const card = data.card;
    state.result = card;
    $('summary').value = card.summary || '';
    $('limitations').value = (card.limitations || []).join('；');
    $('advice1').value = card.recommendations?.[0] ? `${card.recommendations[0].action}。依据：${card.recommendations[0].basis}。风险：${card.recommendations[0].risk}` : '';
    $('advice2').value = card.recommendations?.[1] ? `${card.recommendations[1].action}。依据：${card.recommendations[1].basis}。风险：${card.recommendations[1].risk}` : '';
    $('toolSource').value = [...(card.provenance?.models || []), ...(card.provenance?.tools || [])].join('；');
    state.markers = (card.regions || []).slice(0, 3).map(markerFromRegion);
    renderMarkers(); syncSheet();
    $('modelState').textContent = '待人工复核';
    const relevant = card.regions || [];
    const stats = data.review_stats;
    const counts = stats ? `B 已分级 ${stats.proposed} 处（高 ${stats.grades['高']}、中 ${stats.grades['中']}、低 ${stats.grades['低']}）。` : '';
    setMessage(`${counts}按病害类型与置信度去重后展示 ${relevant.length} 处。`);
    const suggestions = $('modelFindings'); suggestions.replaceChildren();
    relevant.forEach(x => { const p = document.createElement('p'); p.textContent = `${x.region_id} · ${x.disease} · 置信度${x.confidence} · ${x.pending_review || x.confidence === '低' ? '人工复核' : '自动判定'}：${x.observation}`; suggestions.append(p); });
    suggestions.classList.toggle('hidden', !relevant.length);
  } catch (error) { if (state.image !== analyzedImage) return; $('modelState').textContent = '调用失败'; setMessage(error.message, true); }
  finally { if (state.image === analyzedImage) $('analyzeBtn').disabled = false; }
});

function textLine(parent, value, className = '') { const p = document.createElement('p'); p.className = className; p.textContent = value || '待填写'; parent.append(p); }
function syncSheet() {
  $('sheetSummary').textContent = $('summary').value || '待填写总体观察摘要。';
  $('sheetLimitations').textContent = $('limitations').value || '待填写不确定性与人工复核提示。';
  const findings = $('sheetFindings'); findings.replaceChildren();
  state.markers.forEach((m, i) => { const block = document.createElement('div'); block.className = 'sheet-finding'; textLine(block, `${i+1}. ${m.phenomenon || '待填写观察现象'}${m.gradeLabel ? ' · ' + m.gradeLabel : ''}${m.status === '人工复核' && m.gradeLabel?.includes('人工复核') ? '' : '（' + m.status + '）'}`, 'sheet-finding-title'); textLine(block, `依据：${m.evidence || '待填写原图依据'}`); findings.append(block); });
  const advice = $('sheetAdvice'); advice.replaceChildren();
  [$('advice1').value, $('advice2').value].filter(Boolean).forEach((x, i) => textLine(advice, `${i+1}. ${x}`));
  if (!advice.children.length) textLine(advice, '待填写建议、依据与风险。');
  $('sheetImageSource').textContent = `图片：${$('imageSource').value || '待填写来源'}；授权：${$('imageLicense').value || '待填写授权范围'}`;
  $('sheetSceneNote').textContent = `场景说明：${$('sceneNote').value || '无'}`;
  $('sheetToolSource').textContent = `AI／工具：${$('toolSource').value || '待填写模型与工具来源'}`;
  $('sheetReferenceSource').textContent = `公开资料：${$('referenceSource').value || '无'}`;
  $('sheetRunInfo').textContent = state.runId ? `模型运行编号：${state.runId}` : '未调用模型；观察内容由人工填写。';
  $('sheetReviewState').textContent = $('reviewConfirmed').checked ? '人工复核：已确认本卡内容。' : '人工复核：待确认。';
  renderMarkers();
}
fields.forEach(id => $(id).addEventListener('input', syncSheet));
$('reviewConfirmed').addEventListener('change', syncSheet);
$('printBtn').addEventListener('click', () => {
  if (!state.image) { setMessage('请先上传图片。', true); return; }
  if (!$('imageSource').value.trim() || !$('imageLicense').value.trim()) { setMessage('请先填写图片来源与授权范围。', true); return; }
  if (!$('reviewConfirmed').checked) { setMessage('请先对照原图复核，并勾选人工复核确认。', true); return; }
  syncSheet(); window.print();
});
syncSheet();
