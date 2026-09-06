/* ================= SHARED STATE ================= */
const kbDocuments = [
  {name:'Patents Act 1970', type:'LEGISLATION', tier:1, jurisdiction:'Central Act', chunks:412, status:'indexed', date:'2026-09-02'},
  {name:'Patents Rules 2003', type:'RULE', tier:1, jurisdiction:'Central Act', chunks:268, status:'indexed', date:'2026-09-02'},
  {name:'Trade Marks Act 1999', type:'LEGISLATION', tier:1, jurisdiction:'Central Act', chunks:301, status:'indexed', date:'2026-09-02'},
  {name:'Designs Act 2000', type:'LEGISLATION', tier:1, jurisdiction:'Central Act', chunks:154, status:'indexed', date:'2026-09-02'},
  {name:'Copyright Act 1957', type:'LEGISLATION', tier:1, jurisdiction:'Central Act', chunks:288, status:'indexed', date:'2026-09-02'},
  {name:'Geographical Indications Rules 2002', type:'RULE', tier:2, jurisdiction:'Central Act', chunks:97, status:'indexed', date:'2026-09-02'},
  {name:'Biological Diversity Act 2002', type:'LEGISLATION', tier:1, jurisdiction:'Central Act', chunks:88, status:'indexed', date:'2026-09-02'},
  {name:'Biological Diversity Rules 2004', type:'RULE', tier:1, jurisdiction:'Central Act', chunks:74, status:'indexed', date:'2026-09-02'},
  {name:'ABS Guidelines 2014', type:'GUIDELINE', tier:2, jurisdiction:'Central Act', chunks:61, status:'indexed', date:'2026-09-02'},
  {name:'NBA Form I & II', type:'REGISTRY_RECORD', tier:3, jurisdiction:'Central Act', chunks:22, status:'indexed', date:'2026-09-02'},
  {name:'Convention on Biological Diversity (1992)', type:'TREATY', tier:1, jurisdiction:'International', chunks:229, status:'indexed', date:'2026-09-02'},
  {name:'Nagoya Protocol 2010', type:'TREATY', tier:1, jurisdiction:'International', chunks:41, status:'indexed', date:'2026-09-02'},
  {name:'WIPO PCT (notes)', type:'TREATY', tier:2, jurisdiction:'International', chunks:18, status:'indexed', date:'2026-09-02'},
  {name:'WTO TRIPS Agreement (notes)', type:'TREATY', tier:1, jurisdiction:'International', chunks:26, status:'indexed', date:'2026-09-02'},
  {name:'FSSAI Act 2006', type:'LEGISLATION', tier:1, jurisdiction:'Central Act', chunks:63, status:'indexed', date:'2026-09-02'},
];

function renderFiles(){
  const q = (document.getElementById('fileSearch').value || '').toLowerCase();
  const body = document.getElementById('filesBody');
  const rows = kbDocuments.filter(d=>d.name.toLowerCase().includes(q));
  body.innerHTML = rows.map(d=>`
    <tr>
      <td>${d.name}</td>
      <td class="mono" style="color:var(--text-soft);font-size:11px;">${d.type}</td>
      <td><span class="tier-chip t${d.tier}">TIER ${d.tier}</span></td>
      <td style="color:var(--text-soft);">${d.jurisdiction}</td>
      <td class="mono">${d.chunks}</td>
      <td><span class="status-chip ${d.status==='processing'?'processing':''}">${d.status}</span></td>
      <td style="color:var(--text-soft);">${d.date}</td>
    </tr>`).join('');
  document.getElementById('filesTotal').textContent = kbDocuments.length;
  document.getElementById('filesChunks').textContent = kbDocuments.reduce((a,d)=>a+d.chunks,0).toLocaleString();
  document.getElementById('filesTier1').textContent = kbDocuments.filter(d=>d.tier===1).length;
}
document.getElementById('fileSearch').addEventListener('input', renderFiles);

/* ================= NAVIGATION ================= */
const titles = {
  chat: null,
  dashboard: ["Dashboard","System health, corpus coverage and query activity"],
  files: ["Ingested files","Documents indexed into the retrieval corpus"],
  pipeline: null
};
document.querySelectorAll('.navitem').forEach(item=>{
  item.addEventListener('click', ()=>{
    document.querySelectorAll('.navitem').forEach(i=>i.classList.remove('active'));
    item.classList.add('active');
    const v = item.dataset.view;
    document.querySelectorAll('.view').forEach(sec=>sec.classList.remove('active'));
    document.getElementById('view-'+v).classList.add('active');

    document.getElementById('topbarGeneric').style.display = (v==='pipeline') ? 'none' : 'flex';
    document.getElementById('topbarPipeline').style.display = (v==='pipeline') ? 'flex' : 'none';
    if(titles[v]){
      document.getElementById('topTitle').textContent = titles[v][0];
      document.getElementById('topSub').textContent = titles[v][1];
    }
    if(v==='dashboard' && !window.__chartsBuilt){ buildCharts(); window.__chartsBuilt = true; }
    if(v==='pipeline' && !window.__pipelineBuilt){ buildPipeline(); window.__pipelineBuilt = true; }
    if(v==='files'){ renderFiles(); }
  });
});

/* ================= DASHBOARD CHARTS ================= */
function buildCharts(){
  Chart.defaults.font.family = "IBM Plex Sans";
  Chart.defaults.color = "#8B92A6";
  Chart.defaults.font.size = 11;
  const gridColor = 'rgba(255,255,255,.06)';

  new Chart(document.getElementById('chartLatency'), {
    type:'bar',
    data:{ labels:['Sparse','Dense','Fusion','Rerank','Verify'],
      datasets:[{ data:[38,64,12,91,47], backgroundColor:['#E2684A','#4FC3A1','#D9A73B','#4FC3A1','#8B92A6'], borderRadius:4, barThickness:32 }] },
    options:{ plugins:{legend:{display:false}}, scales:{ y:{beginAtZero:true, grid:{color:gridColor}}, x:{grid:{display:false}} } }
  });
  new Chart(document.getElementById('chartVolume'), {
    type:'bar',
    data:{ labels:['Patent','GI','TM','ABS','TK conflict','Formulation'],
      datasets:[{ data:[132,58,74,41,29,78], backgroundColor:'#D9A73B', borderRadius:4 }] },
    options:{ indexAxis:'y', plugins:{legend:{display:false}}, scales:{ x:{beginAtZero:true, grid:{color:gridColor}}, y:{grid:{display:false}} } }
  });
  new Chart(document.getElementById('chartVerify'), {
    type:'line',
    data:{ labels:['W1','W2','W3','W4','W5','W6','W7','W8'],
      datasets:[{ data:[81,83,85,84,88,89,90,91], borderColor:'#4FC3A1', backgroundColor:'rgba(79,195,161,.12)', fill:true, tension:.35, pointRadius:3, pointBackgroundColor:'#4FC3A1' }] },
    options:{ plugins:{legend:{display:false}}, scales:{ y:{min:70,max:100, grid:{color:gridColor}}, x:{grid:{display:false}} } }
  });
  new Chart(document.getElementById('chartJurisdiction'), {
    type:'bar',
    data:{ labels:['Central Act (India)','State Rules','CBD / Nagoya','WIPO treaties','WTO / TRIPS'],
      datasets:[{ data:[190,64,58,52,29], backgroundColor:'#EBE7DC', borderRadius:4, barThickness:26 }] },
    options:{ plugins:{legend:{display:false}}, scales:{ y:{beginAtZero:true, grid:{color:gridColor}}, x:{grid:{display:false}} } }
  });

  const tiers = [{t:'Tier 1', pct:46}, {t:'Tier 2', pct:28}, {t:'Tier 3', pct:17}, {t:'Tier 4', pct:9}];
  document.getElementById('tierRows').innerHTML = tiers.map(x=>`
    <div class="tier-row"><div class="tier-tag">${x.t}</div><div class="tier-bar-bg"><div class="tier-bar-fill" style="width:${x.pct}%"></div></div><div class="tier-pct">${x.pct}%</div></div>`).join('');

  const cats = [{n:'Patent search', v:32}, {n:'Formulation classification', v:19}, {n:'Trademark conflict', v:18}, {n:'ABS compliance', v:10}, {n:'GI registration', v:14}, {n:'TK / prior art', v:7}];
  document.getElementById('catRows').innerHTML = cats.map((x,i)=>`
    <div class="cat-row"><div class="cat-rank">0${i+1}</div><div class="cat-name">${x.n}</div><div class="cat-bar-bg"><div class="cat-bar-fill" style="width:${x.v*3}%"></div></div><div class="cat-val">${x.v}%</div></div>`).join('');
}

/* ================= PIPELINE ================= */
const COLX = [30, 210, 400, 590, 780];
function mkNode(id,col,row,title,type){
  return {id,col,row,title,type,
    x:COLX[col], y: 20 + row*118,
    status:'ok', rate:(Math.random()*30+8).toFixed(1), lag:(Math.random()*0.6+0.15).toFixed(2), seen:Math.floor(Math.random()*4000+500)};
}
const pnodes = [
  mkNode('s1',0,0,'Statutes & Acts','source'), mkNode('s2',0,1,'Regulatory Rules','source'),
  mkNode('s3',0,2,'International Treaties','source'), mkNode('s4',0,3,'Registry Records','source'),
  mkNode('s5',0,4,'Traditional Knowledge (public)','source'), mkNode('s6',0,5,'News & Filings','source'),

  mkNode('b1',1,0,'raw_statutes','ingest'), mkNode('b2',1,1,'raw_rules','ingest'),
  mkNode('b3',1,2,'raw_treaties','ingest'), mkNode('b4',1,3,'raw_registry','ingest'),
  mkNode('b5',1,4,'raw_tk_public','ingest'), mkNode('b6',1,5,'raw_news','ingest'),

  mkNode('i1',2,0.5,'chunked_corpus','index'), mkNode('i2',2,1.7,'sparse_index (BM25)','index'),
  mkNode('i3',2,2.9,'dense_index (bge-m3)','index'), mkNode('i4',2,4.1,'entity_graph','index'),
  mkNode('i5',2,5.1,'authority_tiers','index'),

  mkNode('g1',3,1,'hybrid_fusion (RRF)','reason'), mkNode('g2',3,2.2,'cross_reranking','reason'),
  mkNode('g3',3,3.4,'citation_generation','reason'), mkNode('g4',3,4.6,'verification','reason'),

  mkNode('m1',4,1,'chat_serving','serve'), mkNode('m2',4,2.2,'dashboard_metrics','serve'),
  mkNode('m3',4,3.4,'audit_log','serve'), mkNode('m4',4,4.6,'api_responses','serve'),
];
const pedges = [
  ['s1','b1'],['s2','b2'],['s3','b3'],['s4','b4'],['s5','b5'],['s6','b6'],
  ['b1','i1'],['b2','i1'],['b3','i1'],['b5','i1'],['b6','i1'],
  ['b4','i4'],['b5','i4'],
  ['b1','i5'],['b2','i5'],['b3','i5'],['b4','i5'],
  ['i1','i2'],['i1','i3'],
  ['i2','g1'],['i3','g1'],['i5','g1'],
  ['g1','g2'],['g2','g3'],['i4','g3'],['i5','g3'],['g3','g4'],
  ['g4','m1'],['g4','m2'],['g1','m2'],['g4','m3'],['g3','m3'],['g4','m4'],
];
let selectedNodeId = null;
let queriesServed = 412;

const nodeDescriptions = {
  source:'External or partner-maintained source. Nothing upstream — this is where the corpus originates.',
  ingest:'Parses the raw document, extracts legal structure (Act → Chapter → Section → Clause), and normalizes encoding before handing off to indexing.',
  index:'Builds the retrievable representation of the corpus — lexical, semantic, entity, or authority-weight indices used by the reasoning layer.',
  reason:'Combines retrieved evidence, reranks it, drafts a citation-first answer, and verifies it against the source before release.',
  serve:'Delivers a verified result to a consumer — chat, dashboard, audit trail, or the public API.'
};

function buildPipeline(){
  const svg = document.getElementById('connectorSvg');
  const canvas = document.getElementById('nodeCanvas');
  const maxY = Math.max(...pnodes.map(n=>n.y)) + 120;
  const maxX = COLX[4] + 140;
  svg.setAttribute('width', maxX); svg.setAttribute('height', maxY);
  canvas.style.height = maxY+'px';

  pnodes.forEach(n=>{
    const div = document.createElement('div');
    div.className = 'pnode'; div.id = 'pn-'+n.id;
    div.style.left = (n.x+59)+'px'; div.style.top = n.y+'px';
    div.innerHTML = `<div class="ring"></div><div class="p-title">${n.title}</div><div class="p-status">OK</div><div class="p-meta">${n.rate}/s &middot; lag ${n.lag}s</div>`;
    div.addEventListener('click', ()=>selectNode(n.id));
    canvas.appendChild(div);
  });

  const NW=118, NH=46;
  let svgInner = '<defs><linearGradient id="flowGrad" x1="0" y1="0" x2="1" y2="0"><stop offset="0%" stop-color="#D9A73B" stop-opacity="0"/><stop offset="50%" stop-color="#D9A73B" stop-opacity="1"/><stop offset="100%" stop-color="#D9A73B" stop-opacity="0"/></linearGradient></defs>';
  pedges.forEach(([a,b])=>{
    const na = pnodes.find(x=>x.id===a), nb = pnodes.find(x=>x.id===b);
    const x1 = na.x+59+23, y1 = na.y+23, x2 = nb.x+59-23, y2 = nb.y+23;
    const midX = (x1+x2)/2;
    const d = `M ${x1} ${y1} C ${midX} ${y1}, ${midX} ${y2}, ${x2} ${y2}`;
    const dur = (2 + Math.random()*1.5).toFixed(2);
    svgInner += `<path id="edge-${a}-${b}" d="${d}" fill="none" stroke="rgba(255,255,255,.10)" stroke-width="1.4"/>
      <circle r="2.6" fill="#D9A73B" opacity=".85"><animateMotion dur="${dur}s" repeatCount="indefinite" path="${d}"/></circle>`;
  });
  svg.innerHTML = svgInner;

  document.getElementById('railSources').innerHTML = pnodes.filter(n=>n.type==='source').map(n=>
    `<div class="src-row" data-id="${n.id}"><span class="sr-name">${n.title}</span><span class="sr-rate">${n.rate}/s</span></div>`).join('');
  document.querySelectorAll('.src-row').forEach(r=>r.addEventListener('click',()=>selectNode(r.dataset.id)));

  document.getElementById('rateSlider').addEventListener('input', e=>{ document.getElementById('rateVal').textContent = parseFloat(e.target.value).toFixed(1)+'x'; });
  document.getElementById('qpmSlider').addEventListener('input', e=>{ document.getElementById('qpmVal').textContent = parseFloat(e.target.value).toFixed(1)+'/min'; });
  document.getElementById('breakBtn').addEventListener('click', breakSelectedNode);
  document.getElementById('healBtn').addEventListener('click', healAll);

  selectNode('s1');
  addLog('system','pipeline','stream started &mdash; 25/25 nodes healthy','ok');
  setInterval(()=>{
    queriesServed += Math.floor(Math.random()*3);
    document.getElementById('statQueries').textContent = 'QUERIES #'+queriesServed;
    document.getElementById('statThroughput').textContent = 'THROUGHPUT '+(38+Math.floor(Math.random()*30))+'/s';
  }, 2200);
  updateHealth();
}

function selectNode(id){
  selectedNodeId = id;
  document.querySelectorAll('.pnode').forEach(d=>d.classList.remove('selected'));
  document.getElementById('pn-'+id).classList.add('selected');
  document.querySelectorAll('svg.connectors path').forEach(p=>p.setAttribute('stroke','rgba(255,255,255,.10)'));
  pedges.forEach(([a,b])=>{ if(a===id||b===id){ const p=document.getElementById(`edge-${a}-${b}`); if(p) p.setAttribute('stroke','#D9A73B'); } });
  renderDetail(id);
}

function upstreamOf(id){ return pedges.filter(([a,b])=>b===id).map(([a])=>pnodes.find(n=>n.id===a).title); }
function downstreamOf(id){ return pedges.filter(([a,b])=>a===id).map(([,b])=>pnodes.find(n=>n.id===b).title); }

function sampleRow(node){
  const jur = ['Central Act','State Rule','International'][Math.floor(Math.random()*3)];
  const tier = 'T'+(Math.floor(Math.random()*3)+1);
  const ref = node.type==='source'||node.type==='ingest'
    ? ['§3(d)','§21','Art. 15','Rule 4','§2(1)(e)'][Math.floor(Math.random()*5)]
    : 'Q-'+Math.floor(Math.random()*90000+10000);
  const ts = new Date(Date.now()-Math.floor(Math.random()*60000)).toLocaleTimeString('en-GB');
  return {ts, ref, jur, tier};
}

function renderDetail(id){
  const n = pnodes.find(x=>x.id===id);
  const rows = Array.from({length:5},()=>sampleRow(n));
  document.getElementById('pDetail').innerHTML = `
    <span class="d-tag">${n.type.toUpperCase()}</span>
    <h2>${n.title}</h2>
    <div class="d-meta">${n.id}.v3 &middot; jsonl &middot; sla ${n.lag}s</div>
    <div class="d-stats">
      <div class="stat"><div class="s-lbl">RATE</div><div class="s-val">${n.rate}/s</div></div>
      <div class="stat"><div class="s-lbl">SEEN</div><div class="s-val">${n.seen.toLocaleString()}</div></div>
      <div class="stat"><div class="s-lbl">LAG</div><div class="s-val">${n.lag}s</div></div>
      <div class="stat"><div class="s-lbl">STATE</div><div class="s-val">${n.status==='ok'?'OK':'FAIL'}</div></div>
    </div>
    <div class="d-state ${n.status!=='ok'?'bad':''}">${n.status==='ok'?'Healthy':'Failing &middot; retrying'}</div>
    <div class="d-desc">${nodeDescriptions[n.type]}</div>
    <div class="d-flow"><div class="f-lbl">READS FROM</div>${upstreamOf(id).map(t=>`<span class="flow-pill">${t}</span>`).join('') || '<span class="flow-pill">external source</span>'}</div>
    <div class="d-flow"><div class="f-lbl">FEEDS</div>${downstreamOf(id).map(t=>`<span class="flow-pill">${t}</span>`).join('') || '<span class="flow-pill">end of pipeline</span>'}</div>
    <div class="d-flow"><div class="f-lbl">SAMPLE ROWS &middot; LIVE</div>
      <table class="sample-table"><thead><tr><th>TS</th><th>REF</th><th>JURIS.</th><th>TIER</th></tr></thead>
      <tbody>${rows.map((r,i)=>`<tr class="${i===0?'hl':''}"><td>${r.ts}</td><td>${r.ref}</td><td>${r.jur}</td><td>${r.tier}</td></tr>`).join('')}</tbody></table>
    </div>
    <div class="pulse-block">
      <div class="pulse-ring">&#9679;</div>
      <div class="pulse-count" id="pulseCount">${n.seen.toLocaleString()}</div>
      <div class="pulse-lbl">LIFETIME PROCESSED</div>
    </div>
  `;
}

function addLog(kind,node,msg,cls){
  const body = document.getElementById('logBody');
  const div = document.createElement('div');
  div.className = 'log-line';
  const ts = new Date().toLocaleTimeString('en-GB');
  div.innerHTML = `<span class="l-ts">${ts}</span><span class="l-node">${node}</span><span class="l-msg ${cls}">${msg}</span>`;
  body.appendChild(div);
  body.scrollTop = body.scrollHeight;
  while(body.children.length > 60){ body.removeChild(body.firstChild); }
}

function updateHealth(){
  const failing = pnodes.filter(n=>n.status==='failing').length;
  document.getElementById('failingCount').textContent = failing;
  document.getElementById('failingCount').classList.toggle('bad', failing>0);
  document.getElementById('queuedCount').textContent = failing*Math.floor(Math.random()*4+1);
  document.getElementById('statNodes').textContent = `NODES ${pnodes.length-failing}/${pnodes.length} HEALTHY`;
}

function breakSelectedNode(){
  if(!selectedNodeId) return;
  const n = pnodes.find(x=>x.id===selectedNodeId);
  if(n.status==='failing') return;
  n.status='failing';
  document.getElementById('pn-'+n.id).classList.add('failing');
  document.querySelector(`#pn-${n.id} .p-status`).textContent='FAILING';
  updateHealth();
  addLog('warn', n.title, 'retry 1/3 failed &mdash; backing off 1.5s', 'bad');
  renderDetail(n.id);
  let attempt=1;
  const seq = setInterval(()=>{
    attempt++;
    if(attempt<=3){
      addLog('warn', n.title, `retry ${attempt}/3 failed &mdash; backing off ${(attempt*1.5).toFixed(1)}s`, 'bad');
    } else {
      clearInterval(seq);
      n.status='ok';
      document.getElementById('pn-'+n.id).classList.remove('failing');
      document.querySelector(`#pn-${n.id} .p-status`).textContent='OK';
      addLog('ok', n.title, `recovered on retry ${attempt-1} &mdash; replaying buffered rows`, 'ok');
      setTimeout(()=>addLog('ok', n.title, 'healthy again &mdash; backlog drained, lag back inside sla', 'ok'), 700);
      updateHealth();
      if(selectedNodeId===n.id) renderDetail(n.id);
    }
  }, 1400);
}
function healAll(){
  pnodes.forEach(n=>{
    if(n.status!=='ok'){
      n.status='ok';
      const el = document.getElementById('pn-'+n.id);
      if(el){ el.classList.remove('failing'); el.querySelector('.p-status').textContent='OK'; }
    }
  });
  updateHealth();
  addLog('ok','system','heal all &mdash; every node forced healthy, backfilling from checkpoint','ok');
  if(selectedNodeId) renderDetail(selectedNodeId);
}

/* ================= CHAT ================= */
const chatMessages = document.getElementById('chatMessages');
const chatInput = document.getElementById('chatInput');
const sendBtn = document.getElementById('sendBtn');

function addUserMsg(text){
  const row = document.createElement('div');
  row.className='msg-row user';
  row.innerHTML = `<div class="bubble-user">${text}</div>`;
  chatMessages.appendChild(row);
  scrollChat();
  return row;
}
function addAssistantMsg(html){
  const row = document.createElement('div');
  row.className='msg-row';
  row.innerHTML = `<div class="avatar">S</div><div class="assistant-body">${html}</div>`;
  chatMessages.appendChild(row);
  scrollChat();
  return row;
}
function addSysNote(text){
  const row = document.createElement('div');
  row.className='msg-row'; row.style.justifyContent='center';
  row.innerHTML = `<div class="sys-note">${text}</div>`;
  chatMessages.appendChild(row);
  scrollChat();
  return row;
}
function scrollChat(){ document.querySelector('.chat-scroll').scrollTop = 999999; }

const canned = [
  {q:'section 3(d)', a:'Section 3(d) of the Patents Act 1970 excludes the mere discovery of a new form of a known substance which does not enhance its known efficacy, unless it results in a new product or uses at least one new reactant.',
   cites:[{n:'1', src:'Patents Act 1970, §3(d)', tier:'Tier 1'}]},
  {q:'geographical indication', a:'A Geographical Indication may be registered under the GI Act 1999 where the quality, reputation or characteristic of the good is essentially attributable to its geographical origin, following the Part A/B registration process before the GI Registry.',
   cites:[{n:'1', src:'GI Act 1999, §2(1)(e)', tier:'Tier 1'},{n:'2', src:'GI Registry — Registered GIs, Part A', tier:'Tier 3'}]},
  {q:'abs', a:'Access and Benefit Sharing obligations under the Biological Diversity Act 2002 require prior approval from the National Biodiversity Authority before accessing biological resources for commercial use, with benefit-sharing terms fixed under the ABS Guidelines 2014.',
   cites:[{n:'1', src:'Biological Diversity Act 2002, §21', tier:'Tier 1'},{n:'2', src:'ABS Guidelines 2014', tier:'Tier 2'}]}
];
let defaultAnswerIdx = 0;
function getAnswer(q){
  const lower = q.toLowerCase();
  const hit = canned.find(c=>lower.includes(c.q));
  if(hit) return hit;
  const fb = canned[defaultAnswerIdx % canned.length]; defaultAnswerIdx++;
  return fb;
}

function runQuery(q){
  addUserMsg(q);
  const stages = ['Classifying intent & jurisdiction','Hybrid retrieval (sparse + dense)','Reranking & evidence selection','Citation-first generation & verification'];
  const row = document.createElement('div');
  row.className='msg-row';
  row.innerHTML = `<div class="avatar">S</div><div class="thinking" id="thinkingBlock">${stages.map(s=>`<div class="t-row"><span class="t-dot"></span>${s}</div>`).join('')}</div>`;
  chatMessages.appendChild(row); scrollChat();
  const dots = row.querySelectorAll('.t-dot'); const trows = row.querySelectorAll('.t-row');
  let step=0;
  const iv = setInterval(()=>{
    if(step<dots.length){ dots[step].classList.add('done'); trows[step].classList.add('done'); step++; scrollChat(); }
    else{
      clearInterval(iv); row.remove();
      const ans = getAnswer(q);
      const citesHtml = ans.cites.map(c=>`<span class="cite" title="${c.src} (${c.tier})">[${c.n}]</span>`).join('');
      addAssistantMsg(`${ans.a}${citesHtml}<div class="src-line">${ans.cites.map(c=>`[${c.n}] ${c.src} &middot; ${c.tier}`).join('<br>')}</div>`);
    }
  }, 420);
}
sendBtn.addEventListener('click', ()=>{ const q=chatInput.value.trim(); if(!q) return; chatInput.value=''; runQuery(q); });
chatInput.addEventListener('keydown', e=>{ if(e.key==='Enter'){ sendBtn.click(); } });

/* ---- file upload / ingestion sim ---- */
const fileInput = document.getElementById('fileInput');
const attachBtn = document.getElementById('attachBtn');
attachBtn.addEventListener('click', ()=>fileInput.click());
fileInput.addEventListener('change', ()=>{ if(fileInput.files.length) ingestFile(fileInput.files[0]); });

function ingestFile(file){
  addSysNote(`Uploading <strong>${file.name}</strong>&hellip;`);
  const stages = ['Parsing document','Extracting legal structure','Chunking','Generating embeddings','Assigning authority tier','Indexing'];
  const row = document.createElement('div');
  row.className='msg-row';
  row.innerHTML = `<div class="avatar">S</div><div class="thinking" id="ingestBlock">${stages.map(s=>`<div class="t-row"><span class="t-dot"></span>${s}</div>`).join('')}</div>`;
  chatMessages.appendChild(row); scrollChat();
  const dots = row.querySelectorAll('.t-dot'); const trows = row.querySelectorAll('.t-row');
  let step=0;
  const iv = setInterval(()=>{
    if(step<dots.length){ dots[step].classList.add('done'); trows[step].classList.add('done'); step++; scrollChat(); }
    else{
      clearInterval(iv); row.remove();
      addSysNote(`<strong>${file.name}</strong> added to the knowledge base &mdash; classified Tier 2, indexed.`);
      kbDocuments.unshift({name:file.name, type:'UPLOADED', tier:2, jurisdiction:'Unclassified', chunks:Math.floor(Math.random()*80+20), status:'indexed', date:new Date().toISOString().slice(0,10)});
    }
  }, 380);
}

renderFiles();
