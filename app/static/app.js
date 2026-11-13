/* FraudScan — demo. JS vanilla, sin dependencias. Todos los puntajes y decisiones vienen de la API
   (modelos entrenados); aquí solo se dibuja. */

const state = {
  view: 'statement', model: 'B', meta: null, samples: [], stats: null, resStale: false,
  stmt: null, stmtErr: '', stmtBusy: false, stmtFile: null, stmtName: '',
  form: null, res: null, formErr: '',
  stream: { rows: [], totals: null, position: 0, total: 0, playing: null, err: '' },
};

const usd = (n, d = 2) => (n < 0 ? '−' : '') + '$' + Math.abs(n).toLocaleString('en-US', { minimumFractionDigits: d, maximumFractionDigits: d });
const esc = (s) => String(s ?? '').replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const pts = (score) => (100 * score).toFixed(1);
const DEC = {
  APROBAR: { cls: 'c-low', color: 'var(--risk-low)', level: 'BAJO' },
  VERIFICAR: { cls: 'c-mid', color: 'var(--risk-mid)', level: 'MEDIO' },
  BLOQUEAR: { cls: 'c-high', color: 'var(--risk-high)', level: 'ALTO' },
};
const LEVEL_COLOR = { ALTO: 'var(--risk-high)', MEDIO: 'var(--risk-mid)', BAJO: 'var(--risk-low)' };
const TYPE = { Credit: 'Crédito', Debit: 'Débito', 'Debit (Prepaid)': 'Prepago' };
const MODEL_NAME = { A: 'Modelo A · MLP', B: 'Modelo B · LSTM' };

const ICON = {
  doc: '<svg width="12" height="14" viewBox="0 0 12 14" aria-hidden="true"><path d="M1 1h7l3 3v9H1z" fill="none" stroke="currentColor" stroke-width="1.4"/></svg>',
  folder: '<svg width="20" height="16" viewBox="0 0 20 16" aria-hidden="true"><path d="M1 3h6l2 2h10v10H1z" fill="#4aa3ff"/></svg>',
  search: '<svg width="14" height="14" viewBox="0 0 14 14" aria-hidden="true"><circle cx="6" cy="6" r="4.2" fill="none" stroke="currentColor" stroke-width="1.5"/><path d="M9.5 9.5L13 13" stroke="currentColor" stroke-width="1.5"/></svg>',
  play: '<svg width="12" height="14" viewBox="0 0 12 14" aria-hidden="true"><path d="M1 1l10 6-10 6z" fill="none" stroke="currentColor" stroke-width="1.4"/></svg>',
  chart: '<svg width="14" height="14" viewBox="0 0 14 14" aria-hidden="true"><path d="M1 13h12M3 13V8M7 13V3M11 13V6" fill="none" stroke="currentColor" stroke-width="1.5"/></svg>',
  plus: '<svg width="14" height="14" viewBox="0 0 14 14" aria-hidden="true"><path d="M2 7h10M7 2v10" stroke="#cfe0ff" stroke-width="1.5" fill="none" stroke-dasharray="2 2"/></svg>',
  chip: '<svg width="38" height="28" viewBox="0 0 38 28" aria-hidden="true"><rect x="1" y="1" width="36" height="26" rx="4" fill="none" stroke="#cfe0ff" stroke-width="1.5"/><path d="M1 10h36M1 18h36M14 1v26M24 1v26" stroke="#cfe0ff" fill="none"/></svg>',
  wave: '<svg width="26" height="26" viewBox="0 0 26 26" aria-hidden="true"><path d="M8 7c3 4 3 8 0 12M13 4c5 6 5 12 0 18M18 1c7 8 7 16 0 24" fill="none" stroke="#cfe0ff" stroke-width="1.6"/></svg>',
};

async function api(path, opts) {
  const r = await fetch(path, opts);
  const j = await r.json().catch(() => ({ detail: 'Respuesta inválida del servidor.' }));
  if (!r.ok) throw new Error(typeof j.detail === 'string' ? j.detail : JSON.stringify(j.detail));
  return j;
}
const post = (path, body) => api(path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });

// ------------------------------------------------------------------ piezas comunes
function cardGraphic(cardId, brand, type, footer) {
  const last4 = String(cardId).replace(/\D/g, '').padStart(4, '0').slice(-4);
  const [left, right] = footer || [`Tarjeta ${cardId}`, brand || TYPE[type] || ''];
  return `<div class="card">
    <div class="top">${ICON.chip}${ICON.wave}</div>
    <div class="num">•••• •••• •••• ${esc(last4)}</div>
    <div class="bot"><span>${esc(left)}</span><span>${esc(right)}</span></div>
  </div>`;
}
function gauge(number, color, label, caption, thresholds, capId) {
  const sc = (x) => 100 * Math.sqrt(Math.min(Math.max(x, 0), 1));   // agranda la zona baja, donde están los umbrales
  const marks = thresholds ? `<span class="mk" style="left:${sc(thresholds.t_verify)}%"></span><span class="mk" style="left:${sc(Math.min(thresholds.t_block, 1))}%"></span>` : '';
  const ticks = thresholds ? `<div class="ticks"><span>Verificar desde ${pts(thresholds.t_verify)}</span><span>Bloquear desde ${pts(thresholds.t_block)}</span></div>` : '';
  return `<div class="gauge" style="color:${color}"><div class="n">${esc(number)}</div><div class="lv">${esc(label)}</div>
    <div class="bar" role="img" aria-label="Puntaje ${esc(number)} de 100"><i style="width:${sc(Number(number) / 100)}%;background:${color}"></i>${marks}</div>
    ${ticks}${caption ? `<div class="cap"${capId ? ` id="${capId}"` : ''}>${esc(caption)}</div>` : ''}</div>`;
}
const decTag = (label) => `<span class="tag ${DEC[label].cls}">${label}</span>`;

// ------------------------------------------------------------------ vista: estado de cuenta
function viewStatement() {
  const s = state.stmt;
  const samples = state.samples.map((x) => `
    <div class="sample"><div class="d"><b>${esc(x.title)}</b><span>${esc(x.description)}</span></div>
      <div class="a"><button class="seg" data-act="sample" data-url="${esc(x.url)}" data-name="${esc(x.file)}">Analizar</button>
      <a class="seg" href="${esc(x.url)}" download="${esc(x.file)}">Descargar PDF</a></div></div>`).join('');
  const drop = `<div class="drop" id="drop">
      <div class="big">${state.stmtBusy ? 'Analizando…' : 'Suelta aquí un estado de cuenta en PDF'}</div>
      <div class="sm">El servicio lee las transacciones, las puntúa en orden con el ${esc(MODEL_NAME[state.model])} y decide si hay fraude.</div>
      <button class="btn2" data-act="pick" ${state.stmtBusy ? 'disabled' : ''}>Elegir archivo</button>
    </div>`;
  const err = state.stmtErr ? `<div class="msg err" role="alert">${esc(state.stmtErr)}</div>` : '';

  if (!s) {
    return `<div class="rowwrap">
      <section class="panel" style="flex:1 1 380px" aria-label="Subir estado de cuenta">
        <header><span>Estado de cuenta</span><small>PDF</small></header>
        <div class="gridbg">${drop}</div>${err}
      </section>
      <section class="panel" style="flex:1 1 340px" aria-label="Estados de cuenta de ejemplo">
        <header><span>Ejemplos del periodo de prueba</span><small>${state.samples.length} PDF</small></header>
        <div class="samples">${samples}</div>
        <div class="hintline" style="padding-top:12px">Cada ejemplo es un mes real de una tarjeta que el modelo nunca vio al entrenar. Puedes descargarlo y volver a subirlo.</div>
      </section>
    </div>`;
  }

  const h = s.header, sum = s.summary, color = LEVEL_COLOR[s.verdict.level];
  const flagged = s.rows.filter((r) => r.decision !== 0);
  const sigs = flagged.length
    ? flagged.map((r) => `<div class="sig"><span>${esc(r.date)} ${esc(r.time)} · ${esc(r.mcc_name)} · ${esc(r.loc)} · ${usd(r.amount)}</span><b class="${DEC[r.decision_label].cls}">${r.decision_label} ${pts(r.score)}</b></div>`).join('')
    : '<div class="sig" style="color:var(--muted)">Ninguna transacción marcada</div>';
  const first = flagged[0];
  const brief = flagged.length
    ? `Se analizaron ${sum.transactions} transacciones por ${usd(sum.total_usd)}. El modelo marcó ${sum.blocked} para bloqueo (${usd(sum.blocked_usd)}) y ${sum.verify} para verificación (${usd(sum.verify_usd)}). La primera transacción marcada es del ${first.date} a las ${first.time}, en ${first.loc}.`
    : `Se analizaron ${sum.transactions} transacciones por ${usd(sum.total_usd)} y ninguna supera el umbral de verificación.`;
  const truth = s.truth ? `
      <div class="kv"><span>Fraudes reales marcados</span><span>${s.truth.frauds_flagged} de ${s.truth.frauds}</span></div>
      <div class="kv"><span>Falsas alarmas</span><span>${s.truth.false_alarms}</span></div>` : '';
  const hasTruth = !!s.truth;
  const rows = s.rows.map((r) => `<tr class="${r.decision !== 0 ? 'hot' : ''}">
      <td class="num">${r.n}</td><td>${esc(r.date)} ${esc(r.time)}</td><td>${esc(r.mcc_name)}</td><td>${esc(r.loc === 'En línea' ? r.loc : `${r.loc} · ${r.channel}`)}</td>
      <td class="num">${usd(r.amount)}</td><td class="num">${pts(r.score)}</td><td>${decTag(r.decision_label)}</td>
      ${hasTruth ? `<td class="${r.is_fraud ? 'c-high' : 'c-muted'}">${r.is_fraud ? 'Fraude' : 'Legítima'}</td>` : ''}
      <td class="wrap">${esc(r.reasons.map((x) => x.label).join(' · ') || (r.errors.join(' · ')))}</td></tr>`).join('');

  return `<div class="rowwrap">
      <section class="panel" style="flex:1 1 360px" aria-label="Estado de cuenta analizado">
        <header><span>Estado de cuenta</span><small>${esc(h.statement_id)}</small></header>
        <div class="gridbg">${cardGraphic(h.card_id, h.brand, h.card_type)}
          <div class="amt"><div class="big">${usd(sum.total_usd)}</div><div class="m">${sum.transactions} transacciones · ${esc(h.home_city)}, ${esc(h.home_state)}</div>
          <div class="t">${esc(h.period_start)} a ${esc(h.period_end)} · ${esc(state.stmtName)}</div></div>
        </div>${err}
      </section>
      <section class="panel" style="flex:1 1 300px" aria-label="Veredicto">
        <header><span>Veredicto</span><small>${esc(MODEL_NAME[s.model])}</small></header>
        ${gauge(Math.round(100 * sum.max_score), color, `Riesgo ${s.verdict.level} · ${s.verdict.title}`, 'Puntaje más alto entre las transacciones del periodo', s.thresholds)}
      </section>
    </div>
    <div class="rowwrap">
      <section class="panel" style="flex:1 1 340px" aria-label="Informe">
        <header><span style="display:flex;align-items:center;gap:8px">${ICON.doc}<span>Informe</span></span></header>
        <div class="brief">${esc(brief)}</div>
        ${s.warnings.map((w) => `<div class="msg warn">${esc(w)}</div>`).join('')}
        <div class="sigs" style="padding-top:12px"><div class="h">Transacciones marcadas</div>${sigs}</div>
      </section>
      <section class="panel" style="flex:0 1 340px;min-width:260px" aria-label="Detalles">
        <header>Detalles</header>
        <div class="kv"><span>Resultado</span><span style="color:${color}">${esc(s.verdict.title)}</span></div>
        <div class="kv"><span>A bloquear</span><span>${sum.blocked} · ${usd(sum.blocked_usd)}</span></div>
        <div class="kv"><span>A verificar</span><span>${sum.verify} · ${usd(sum.verify_usd)}</span></div>
        ${truth}
        <div class="kv"><span>Tiempo de análisis</span><span>${sum.latency_total_ms.toFixed(1)} ms</span></div>
        <div class="reco">${esc(s.verdict.text)}${hasTruth ? ' La comparación con la realidad solo existe para los ejemplos, que traen la etiqueta del dataset.' : ''}</div>
        <div class="grow"></div>
        <div class="actions"><button class="btn" data-act="pick">${ICON.plus}<span>Subir otro PDF</span></button>
          <button class="btn2" data-act="stmt-clear">Volver a los ejemplos</button></div>
      </section>
    </div>
    <section class="panel" aria-label="Transacciones del estado de cuenta">
      <header><span>Transacciones del periodo</span><small>${flagged.length} marcadas de ${sum.transactions}</small></header>
      <div class="tablewrap"><table>
        <thead><tr><th class="num">N.º</th><th>Fecha</th><th>Categoría</th><th>Ubicación</th><th class="num">Monto</th><th class="num">Puntaje</th><th>Decisión</th>${hasTruth ? '<th>Real</th>' : ''}<th>Por qué</th></tr></thead>
        <tbody>${rows}</tbody></table></div>
    </section>`;
}

// ------------------------------------------------------------------ vista: una transacción
const STALE_CAP = 'Resultado de la compra anterior · vuelve a autorizar';

// Marca el puntaje como viejo cuando el formulario cambia después de autorizar.
function updatePreview() {
  if (state.view !== 'transaction' || !state.res) return;
  const panel = document.getElementById('gaugePanel');
  if (panel) panel.classList.toggle('stale', !!state.resStale);
  const cap = document.getElementById('gaugeCap');
  if (cap) cap.textContent = state.resStale ? STALE_CAP : `Riesgo ${DEC[state.res.decision_label].level}`;
}

function viewTransaction() {
  const m = state.meta, f = state.form, r = state.res;
  const card = m.cards.find((c) => c.id === f.card) || m.cards[0];
  const online = f.channel === 'Online Transaction';
  const opt = (v, label, sel) => `<option value="${esc(v)}" ${String(v) === String(sel) ? 'selected' : ''}>${esc(label)}</option>`;
  const locs = online ? opt('ONLINE', 'En línea', 'ONLINE')
    : `<optgroup label="Estados Unidos">${m.locations.filter((l) => !l.foreign && l.value !== 'ONLINE').map((l) => opt(l.value, l.label, f.loc)).join('')}</optgroup>
       <optgroup label="Otros países">${m.locations.filter((l) => l.foreign).map((l) => opt(l.value, l.label, f.loc)).join('')}</optgroup>`;
  const seg = (k, v, label) => `<button class="seg" data-act="set" data-k="${k}" data-v="${esc(v)}" aria-pressed="${f[k] === v}">${esc(label)}</button>`;

  let g, brief, sigs, details;
  if (r) {
    const d = DEC[r.decision_label];
    g = gauge(pts(r.score), d.color, r.decision_label, state.resStale ? STALE_CAP : `Riesgo ${d.level}`,
              r.thresholds, 'gaugeCap').replace('class="gauge"', 'class="gauge compact"');
    const c = r.context;
    brief = r.decision === 0
      ? 'El modelo no ve señales suficientes para detener esta compra: se aprueba sin fricción para el titular.'
      : r.decision === 1
        ? 'El puntaje cae en la zona dudosa. En lugar de bloquear, se pide una verificación al titular: cuesta poco y evita molestar de más a un cliente legítimo.'
        : 'El puntaje supera el umbral de bloqueo: el riesgo de fraude es tan alto que conviene detener la compra y contactar al titular.';
    sigs = r.reasons.length ? r.reasons.map((x) => `<div class="sig"><span>${esc(x.label)}</span><b class="c-high">+${x.points.toFixed(1)}</b></div>`).join('')
      : `<div class="sig" style="color:var(--muted)">${r.decision === 0 ? 'Sin señales relevantes' : 'Ninguna señal domina: es la combinación'}</div>`;
    const other = Object.entries(r.other_models).map(([k, v]) => `<div class="kv"><span>${esc(MODEL_NAME[k])}</span><span>${['APROBAR', 'VERIFICAR', 'BLOQUEAR'][v.decision]} · ${pts(v.score)}</span></div>`).join('');
    details = `
      <div class="kv"><span>Decisión</span><span style="color:${d.color}">${r.decision_label}</span></div>
      <div class="kv"><span>Puntaje</span><span>${pts(r.score)} / 100</span></div>
      <div class="kv"><span>Latencia</span><span>${r.latency_ms.toFixed(2)} ms</span></div>
      <div class="kv"><span>Entra al historial</span><span>${r.committed ? 'Sí' : 'No, se bloqueó'}</span></div>
      ${other}
      <div class="kv"><span>Monto frente a lo habitual</span><span>${c.amount_vs_usual}×</span></div>
      <div class="kv"><span>Compras en 24 h</span><span>${c.tx_last_24h}</span></div>
      <div class="kv"><span>Ubicación nueva</span><span>${c.new_location ? 'Sí' : 'No'}</span></div>`;
  } else {
    g = gauge('–', 'var(--muted)', 'Sin analizar', 'Define la compra y presiona Autorizar', null, 'gaugeCap')
      .replace('class="gauge"', 'class="gauge compact"');
    brief = 'Define una compra para la tarjeta elegida. El servicio calcula las variables con su historial, la puntúa con el modelo y decide si aprobar, verificar o bloquear.';
    sigs = '<div class="sig" style="color:var(--muted)">Sin señales todavía</div>';
    details = '<div class="kv"><span>Decisión</span><span>Pendiente</span></div>';
  }
  const presets = ['Compra habitual', 'Compra grande en línea', 'En el extranjero', 'En línea con CVV incorrecto']
    .map((p, i) => `<button class="seg" data-act="preset" data-i="${i}">${p}</button>`).join('');

  return `<div class="rowwrap">
      <section class="panel" style="flex:2 1 440px" aria-label="Condiciones de la transacción">
        <header><span>Condiciones de la transacción</span><small>${esc(MODEL_NAME[state.model])}</small></header>
        <div class="form">
          <div class="grp"><span class="lab">Tarjeta</span><div class="opts"><select class="field" data-f="card" aria-label="Tarjeta">${m.cards.map((c) => opt(c.id, `${c.id} · ${c.brand} · ${c.home_state}`, f.card)).join('')}</select></div></div>
          <div class="grp"><span class="lab">Monto (USD)</span><div class="opts"><input class="field" data-f="amount" type="number" step="0.01" value="${esc(f.amount)}" aria-label="Monto en dólares"></div></div>
          <div class="grp"><span class="lab">Minutos desde la anterior</span><div class="opts"><input class="field" data-f="minutes" type="number" min="0" value="${esc(f.minutes)}" aria-label="Minutos desde la compra anterior"></div></div>
          <div class="grp" role="group" aria-label="Canal"><span class="lab">Canal</span><div class="opts">${m.channels.map((c) => seg('channel', c.value, c.label)).join('')}</div></div>
          <div class="grp"><span class="lab">Ubicación</span><div class="opts"><select class="field" data-f="loc" aria-label="Ubicación del comercio" ${online ? 'disabled' : ''}>${locs}</select></div></div>
          <div class="grp"><span class="lab">Categoría</span><div class="opts"><select class="field" data-f="mcc" aria-label="Categoría del comercio">${m.mccs.map((x) => opt(x.code, `${x.name} (${x.code})`, f.mcc)).join('')}</select></div></div>
          <div class="grp"><span class="lab">Incidencia</span><div class="opts"><select class="field" data-f="error" aria-label="Error en la transacción">${opt('', 'Ninguna', f.error)}${m.errors.map((e) => opt(e.value, e.label, f.error)).join('')}</select></div></div>
          <div class="grp" role="group" aria-label="Casos rápidos"><span class="lab">Casos rápidos</span><div class="opts">${presets}</div></div>
        </div>
        ${state.formErr ? `<div class="msg err" role="alert">${esc(state.formErr)}</div>` : ''}
      </section>
      <div class="col" style="flex:1 1 300px">
        <section class="panel ${r && state.resStale ? 'stale' : ''}" id="gaugePanel" aria-label="Puntaje de riesgo">
          <header>Puntaje de riesgo</header>${g}
        </section>
        <section class="panel" aria-label="Tarjeta">
          <header><span>Tarjeta</span><small>${esc(card.id)}</small></header>
          <div class="gridbg compact">${cardGraphic(card.id, card.brand, card.card_type,
            [card.brand || '', TYPE[card.card_type] || card.card_type])}</div>
          <div class="kv compact"><span>Residencia</span><span>${esc(card.home_city)}, ${esc(card.home_state)}</span></div>
          <div class="kv compact"><span>Límite</span><span>${usd(card.credit_limit, 0)}</span></div>
          <div class="kv compact"><span>Compras previas</span><span>${card.prior_transactions.toLocaleString('en-US')}</span></div>
          <div class="kv compact"><span>Gasto habitual</span><span>${card.usual_amount == null ? '–' : usd(card.usual_amount)}</span></div>
        </section>
      </div>
    </div>
    <div class="rowwrap" style="flex:1">
      <section class="panel" style="flex:1 1 340px" aria-label="Informe">
        <header><span style="display:flex;align-items:center;gap:8px">${ICON.doc}<span>Informe</span></span></header>
        <div class="brief">${esc(brief)}</div>
        <div class="sigs"><div class="h">Señales que suben el puntaje</div>${sigs}</div>
      </section>
      <section class="panel" style="flex:0 1 340px;min-width:260px" aria-label="Detalles">
        <header>Detalles</header>${details}
        <div class="grow"></div>
        <div class="actions" style="padding-top:12px"><button class="btn" data-act="score">${ICON.plus}<span>Autorizar</span></button></div>
      </section>
    </div>`;
}

// ------------------------------------------------------------------ vista: flujo de prueba
function viewStream() {
  const st = state.stream, m = state.meta;
  const zero = { value_usd: 0, fraud_stopped_usd: 0, fraud_missed_usd: 0, verified: 0, blocked: 0, legit_bothered: 0, frauds_caught: 0, frauds_missed: 0 };
  const names = { none: 'Sin modelo · aprobar todo', A: MODEL_NAME.A, B: MODEL_NAME.B };
  const totals = ['none', 'A', 'B'].filter((k) => k === 'none' || m.models[k]).map((k) => {
    const t = (st.totals && st.totals[k]) || zero;
    return `<section class="panel total ${k === state.model ? 'sel' : ''}" aria-label="${names[k]}">
      <header><span>${names[k]}</span>${k === state.model ? '<small>Decide</small>' : ''}</header>
      <div class="v">${usd(t.value_usd, 0)}</div><div class="s">de valor frente a no tener modelo</div>
      <div class="s">Fraude detenido: ${usd(t.fraud_stopped_usd, 0)} (${t.frauds_caught} casos) · pasó: ${usd(t.fraud_missed_usd, 0)} (${t.frauds_missed})</div>
      <div class="s">Legítimos molestados: ${t.legit_bothered} · verificaciones ${t.verified} · bloqueos ${t.blocked}</div></section>`;
  }).join('');
  const done = st.total > 0 && st.position >= st.total;
  const rows = st.rows.length ? st.rows.map((r) => {
    const others = Object.entries(r.decisions).filter(([k]) => k !== r.model && k !== 'none').map(([k, v]) => `<span class="tag ${DEC[v].cls}">${k}: ${v}</span>`).join(' ');
    return `<tr class="${r.is_fraud ? 'hot' : ''}"><td>${esc(r.timestamp)}</td><td class="mono">${esc(r.card_id)}</td><td class="num">${usd(r.amount)}</td>
      <td>${esc(r.loc === 'En línea' ? r.loc : `${r.loc} · ${r.channel}`)}</td><td class="num">${pts(r.score)}</td>
      <td>${decTag(r.decision_label)}</td><td class="${r.is_fraud ? 'c-high' : 'c-muted'}">${r.is_fraud ? 'Fraude' : 'Legítima'}</td><td>${others}</td><td>${esc(r.mcc_name)}</td></tr>`;
  }).join('') : '<tr><td colspan="9" class="c-muted">Presiona Reproducir para empezar.</td></tr>';

  return `<div class="rowwrap">${totals}</div>
    <section class="panel" aria-label="Flujo de transacciones del periodo de prueba">
      <header><span>Transacciones no vistas en entrenamiento</span><small>${st.position} de ${m.stream.total} · ${m.stream.frauds} fraudes · ${esc(m.stream.year)}</small></header>
      <div class="actions inline">
        <button class="btn2" data-act="play" ${done ? 'disabled' : ''}>${st.playing ? 'Pausar' : 'Reproducir'}</button>
        <button class="btn2" data-act="step" ${done ? 'disabled' : ''}>Avanzar 10</button>
        <button class="btn2" data-act="all" ${done ? 'disabled' : ''}>Todo</button>
        <button class="btn2" data-act="reset">Reiniciar</button>
      </div>
      <div class="progress"><i style="width:${st.total ? (100 * st.position / st.total) : 0}%"></i></div>
      ${st.err ? `<div class="msg err" role="alert">${esc(st.err)}</div>` : ''}
      <div class="tablewrap"><table>
        <thead><tr><th>Fecha</th><th>Tarjeta</th><th class="num">Monto</th><th>Ubicación</th><th class="num">Puntaje</th><th>Decisión</th><th>Real</th><th>Otro modelo</th><th>Categoría</th></tr></thead>
        <tbody>${rows}</tbody></table></div>
    </section>
    <div class="foot">La etiqueta real se revela después de decidir, para sumar el valor en dólares. El flujo es una selección de 16 tarjetas con muchos casos de fraude; los números representativos están en Resultados.</div>`;
}

// ------------------------------------------------------------------ vista: resultados
function viewResults() {
  const m = state.meta, c = m.costs, cmp = m.comparison_B_vs_A;
  const f = (v, d = 3) => (v == null ? '–' : v.toFixed(d));
  const rows = m.iterations.map((r) => `<tr><td>${esc(r.iteracion)}</td><td class="num">${usd(r.valor_test_usd, 0)}</td><td class="num">${(100 * r.pct_maximo).toFixed(1)} %</td>
    <td class="num">${usd(r.valor_neto_mes_usd, 0)}</td><td class="num">${f(r.recall_test)}</td><td class="num">${f(r.precision_test)}</td><td class="num">${f(r.ap_test)}</td>
    <td class="num">${f(r.latencia_p50_ms, 2)} ms</td><td class="num">${usd(r.valor_ic95_bajo, 0)} a ${usd(r.valor_ic95_alto, 0)}</td><td class="wrap">${esc(r.cambio)}</td></tr>`).join('');
  return `<section class="panel" aria-label="Resultados por iteración">
      <header><span>Resultados por iteración</span><small>Conjunto de prueba 2018–2019</small></header>
      <div class="tablewrap" style="max-height:none"><table>
        <thead><tr><th>Iteración</th><th class="num">Valor (USD)</th><th class="num">% del máximo</th><th class="num">Valor / mes</th><th class="num">Recall</th><th class="num">Precisión</th><th class="num">AP</th><th class="num">Latencia</th><th class="num">IC 95 % del valor</th><th>Cambio</th></tr></thead>
        <tbody>${rows}</tbody></table></div>
    </section>
    <div class="rowwrap">
      <section class="panel" style="flex:1 1 340px" aria-label="Lectura de los resultados">
        <header><span style="display:flex;align-items:center;gap:8px">${ICON.doc}<span>Informe</span></span></header>
        <div class="brief">Los umbrales se eligen por valor en validación (2016–2017) y los resultados se miden en prueba (2018–2019).${cmp ? ` El modelo B aporta ${usd(cmp.extra_value_per_month_usd, 0)} más al mes que el modelo A, con ${cmp.extra_latency_ms.toFixed(2)} ms más de latencia y ${usd(cmp.extra_compute_per_month_usd, 4)} más de cómputo al mes.` : ''}</div>
        <div class="reco">${m.profile === 'quick' ? 'Entrenamiento rápido con una muestra reducida.' : 'Entrenamiento con el perfil completo.'} El dataset es sintético y los costos son supuestos, así que los dólares son una estimación de orden de magnitud.</div>
      </section>
      <section class="panel" style="flex:0 1 380px;min-width:260px" aria-label="Supuestos del modelo de valor">
        <header>Supuestos de costo (USD)</header>
        <div class="kv"><span>Contracargo por fraude</span><span>${usd(c.chargeback_fee)}</span></div>
        <div class="kv"><span>Costo de verificar</span><span>${usd(c.verification_cost)}</span></div>
        <div class="kv"><span>Fricción al verificar</span><span>${usd(c.friction_cost_verify)}</span></div>
        <div class="kv"><span>Fricción al bloquear</span><span>${usd(c.friction_cost_block)}</span></div>
        <div class="kv"><span>Efectividad de verificar</span><span>${(100 * c.otp_effectiveness).toFixed(0)} %</span></div>
      </section>
    </div>`;
}

// ------------------------------------------------------------------ marco
function render() {
  const app = document.getElementById('app');
  if (!state.meta) { app.innerHTML = '<div class="shell"><div class="win"><div class="brief">Cargando…</div></div></div>'; return; }
  const views = [
    ['statement', 'Estado de cuenta', ICON.doc, state.stmt ? state.stmt.verdict.level : 'PDF'],
    ['transaction', 'Analizar transacción', ICON.search, state.res ? state.res.decision_label : ''],
    ['stream', 'Flujo de prueba', ICON.play, state.stream.total ? `${state.stream.position}/${state.stream.total}` : ''],
    ['results', 'Resultados del modelo', ICON.chart, ''],
  ];
  const nav = views.map(([id, label, icon, tag]) => `<button class="row" data-act="view" data-id="${id}" aria-pressed="${state.view === id}"><span class="l">${icon}<span>${label}</span></span>${tag ? `<span class="tag">${esc(tag)}</span>` : ''}</button>`).join('');
  const models = Object.keys(state.meta.models).map((k) => `<button class="row" data-act="model" data-id="${k}" aria-pressed="${state.model === k}"><span class="l"><span>${esc(MODEL_NAME[k])}</span></span>${state.model === k ? '<span class="tag">Decide</span>' : ''}</button>`).join('');
  const s = state.stats || { scored: 0, latency_p50_ms: 0, latency_p95_ms: 0 };
  const body = { statement: viewStatement, transaction: viewTransaction, stream: viewStream, results: viewResults }[state.view]();

  app.innerHTML = `
  <div class="shell"><div class="win">
    <div class="titlebar">
      <div class="t"><svg width="14" height="14" viewBox="0 0 14 14" aria-hidden="true"><rect x="1" y="1" width="12" height="12" fill="none" stroke="#cfe0ff" stroke-width="1.5"/><rect x="4" y="4" width="6" height="6" fill="#cfe0ff"/></svg><span>FraudScan · Centro de autorización</span></div>
      <div class="dots" aria-hidden="true"><span style="background:#f2b84b"></span><span style="background:#4aa3ff"></span><span style="background:#e0524d"></span></div>
    </div>
    <div class="layout">
      <nav class="side" aria-label="Secciones">
        <h2>${ICON.folder}<span>Centro de análisis</span></h2>
        <div class="list">${nav}</div>
        <div class="sep"></div>
        <h3>Modelo que decide</h3>
        <div class="list">${models}</div>
        <div class="grow"></div>
        <div class="stats">
          <div><span>Transacciones puntuadas</span><b>${s.scored.toLocaleString('en-US')}</b></div>
          <div><span>Latencia p50</span><b>${s.scored ? s.latency_p50_ms.toFixed(2) + ' ms' : '–'}</b></div>
          <div><span>Latencia p95</span><b>${s.scored ? s.latency_p95_ms.toFixed(2) + ' ms' : '–'}</b></div>
        </div>
      </nav>
      <main class="main">${body}</main>
    </div>
  </div></div>`;
}
function rerender() { const y = window.scrollY; render(); window.scrollTo(0, y); }

// ------------------------------------------------------------------ acciones
async function refreshStats() { try { state.stats = await api('/api/stats'); } catch (e) { /* se queda el valor anterior */ } }

async function analyzePdf(file, name) {
  state.stmtBusy = true; state.stmtErr = ''; rerender();
  try {
    state.stmt = await api(`/api/statement?model=${encodeURIComponent(state.model)}`, { method: 'POST', headers: { 'Content-Type': 'application/pdf' }, body: file });
    state.stmtFile = file; state.stmtName = name;
  } catch (e) { state.stmtErr = e.message; }
  state.stmtBusy = false;
  await refreshStats(); rerender();
}
function readForm() {
  document.querySelectorAll('[data-f]').forEach((el) => { state.form[el.dataset.f] = el.value; });
}
async function scoreForm() {
  readForm(); state.formErr = '';
  const f = state.form;
  try {
    const amount = parseFloat(f.amount);
    if (!Number.isFinite(amount)) throw new Error('Escribe un monto válido.');
    state.resStale = false;
    state.res = await post('/api/score', { card_id: f.card, amount, channel: f.channel, mcc: parseInt(f.mcc, 10), loc: f.channel === 'Online Transaction' ? 'ONLINE' : f.loc,
      errors: f.error ? [f.error] : [], minutes_after_last: Math.max(0, parseInt(f.minutes || '0', 10)), model: state.model });
    const i = state.meta.cards.findIndex((c) => c.id === state.res.card.id);
    if (i >= 0) state.meta.cards[i] = state.res.card;
  } catch (e) { state.formErr = e.message; }
  await refreshStats(); rerender();
}
function applyPreset(i) {
  readForm();
  const f = state.form, c = state.meta.cards.find((x) => x.id === f.card);
  const usual = c.usual_amount || 40;
  const p = [
    { amount: Math.max(5, usual).toFixed(2), channel: c.has_chip ? 'Chip Transaction' : 'Swipe Transaction', loc: c.home_state, mcc: 5411, minutes: 300, error: '' },
    { amount: (Math.max(40, usual) * 12).toFixed(2), channel: 'Online Transaction', loc: 'ONLINE', mcc: 5732, minutes: 4, error: '' },
    { amount: '310.00', channel: 'Swipe Transaction', loc: 'Italy', mcc: 5311, minutes: 20, error: '' },
    { amount: '189.90', channel: 'Online Transaction', loc: 'ONLINE', mcc: 5815, minutes: 2, error: 'Bad CVV' },
  ][i];
  Object.assign(f, p);
}
async function streamStep(n) {
  const st = state.stream;
  try {
    const r = await post('/api/stream/next', { n, model: state.model });
    st.rows = r.rows.reverse().concat(st.rows); st.totals = r.totals; st.position = r.position; st.total = r.total; st.err = '';
    state.stats = r.stats;
    if (r.position >= r.total) stopPlay();
  } catch (e) { st.err = e.message; stopPlay(); }
  rerender();
}
function stopPlay() { if (state.stream.playing) { clearInterval(state.stream.playing); state.stream.playing = null; } }

document.addEventListener('click', async (e) => {
  const el = e.target.closest('[data-act]');
  if (!el) return;
  const act = el.dataset.act;
  if (act === 'view') { if (state.view === 'transaction') readForm(); state.view = el.dataset.id; rerender(); }
  else if (act === 'model') {
    if (state.view === 'transaction') readForm();
    state.model = el.dataset.id; state.res = null;
    if (state.stmtFile) await analyzePdf(state.stmtFile, state.stmtName); else rerender();
  }
  else if (act === 'pick') document.getElementById('pdfInput').click();
  else if (act === 'sample') {
    try { const blob = await (await fetch(el.dataset.url)).blob(); await analyzePdf(blob, el.dataset.name); }
    catch (err) { state.stmtErr = 'No se pudo cargar el ejemplo.'; rerender(); }
  }
  else if (act === 'stmt-clear') { state.stmt = null; state.stmtFile = null; state.stmtErr = ''; rerender(); }
  else if (act === 'set') { readForm(); state.resStale = true; state.form[el.dataset.k] = el.dataset.v;
    if (el.dataset.k === 'channel' && el.dataset.v !== 'Online Transaction' && state.form.loc === 'ONLINE') state.form.loc = (state.meta.cards.find((c) => c.id === state.form.card) || {}).home_state;
    rerender(); }
  else if (act === 'preset') { applyPreset(Number(el.dataset.i)); state.resStale = true; rerender(); }
  else if (act === 'score') await scoreForm();
  else if (act === 'play') { if (state.stream.playing) { stopPlay(); rerender(); } else { state.stream.playing = setInterval(() => streamStep(3), 450); rerender(); } }
  else if (act === 'step') await streamStep(10);
  else if (act === 'all') { stopPlay(); await streamStep(500); }
  else if (act === 'reset') {
    stopPlay(); await post('/api/reset', {});
    state.stream = { rows: [], totals: null, position: 0, total: 0, playing: null, err: '' }; state.res = null;
    state.meta = await api('/api/meta'); await refreshStats(); rerender();
  }
});
document.addEventListener('input', (e) => {
  if (!e.target.dataset || !e.target.dataset.f) return;
  readForm();
  if (state.res) state.resStale = true;
  updatePreview();
});
document.addEventListener('change', (e) => {
  if (e.target.id === 'pdfInput') { const file = e.target.files[0]; e.target.value = ''; if (file) analyzePdf(file, file.name); return; }
  if (e.target.dataset && e.target.dataset.f && e.target.dataset.f !== 'card') {
    readForm(); if (state.res) state.resStale = true; updatePreview(); return;
  }
  if (e.target.dataset && e.target.dataset.f === 'card') { readForm(); state.res = null; state.resStale = false;
    if (state.form.channel !== 'Online Transaction') state.form.loc = state.meta.cards.find((c) => c.id === state.form.card).home_state;
    rerender(); }
});
['dragenter', 'dragover'].forEach((ev) => document.addEventListener(ev, (e) => { const d = e.target.closest && e.target.closest('#drop'); if (d) { e.preventDefault(); d.classList.add('over'); } }));
document.addEventListener('dragleave', (e) => { const d = e.target.closest && e.target.closest('#drop'); if (d && !d.contains(e.relatedTarget)) d.classList.remove('over'); });
// Soltar un archivo fuera de la zona no debe hacer que el navegador abra el PDF.
window.addEventListener('dragover', (e) => e.preventDefault());
document.addEventListener('drop', (e) => {
  e.preventDefault();
  const d = e.target.closest && e.target.closest('#drop');
  if (!d) return;
  d.classList.remove('over');
  const file = e.dataTransfer.files[0];
  if (file) analyzePdf(file, file.name);
});

async function init() {
  render();
  try {
    [state.meta, state.samples] = await Promise.all([api('/api/meta'), api('/api/statement/samples')]);
    if (!state.meta.models[state.model]) state.model = Object.keys(state.meta.models)[0];
    const c = state.meta.cards[0];
    state.form = { card: c.id, amount: '42.50', minutes: '180', channel: 'Chip Transaction', loc: c.home_state, mcc: '5411', error: '' };
    state.stream.position = state.meta.stream.position; state.stream.total = state.meta.stream.total;
    await refreshStats();
  } catch (e) {
    document.getElementById('app').innerHTML = `<div class="shell"><div class="win"><div class="msg err" role="alert">No se pudo cargar el demo: ${esc(e.message)}</div></div></div>`;
    return;
  }
  render();
}
init();
