import React, { useState, useEffect, useRef } from 'react';
import {
  LineChart, Line, AreaChart, Area, BarChart, Bar,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine
} from 'recharts';

// =============================================================================
// SITARAM HFT Dashboard v3.0
// Data source: polls http://localhost:8001/metrics every 2s via fetch
// Fields match claude-agent /metrics endpoint exactly
// =============================================================================

const METRICS_URL = 'http://localhost:8001/metrics';
const ALERTS_URL  = 'http://localhost:8001/alerts/recent';
const POLL_MS     = 2000;
const ALERT_MS    = 10000;
const MAX_PTS     = 60;

const C = {
  bg:    '#060912', panel:  '#0c1220', panel2: '#0f1828',
  border:'#1a2540', green:  '#00e676', red:    '#ff3d5a',
  amber: '#ffb300', blue:   '#29b6f6', purple: '#ce93d8',
  teal:  '#26c6da', dim:    '#4a6080', sub:    '#7a9abf',
  text:  '#cde0f5',
};

const push  = (arr, item, max = MAX_PTS) => [...arr.slice(-(max-1)), item];
const fmt2  = v => (v >= 0 ? '+' : '') + Number(v).toFixed(2);
const fmtP  = v => Number(v).toFixed(2) + '%';
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
const TT    = {
  contentStyle: { background:'#0c1220', border:'1px solid #1a2540',
                  color:'#cde0f5', fontSize:10 },
};

// ── Shared components ────────────────────────────────────────────────────────
function Panel({ title, badge, accent = C.blue, children, style = {} }) {
  return (
    <div style={{
      background:`linear-gradient(145deg,${C.panel},${C.panel2})`,
      border:`1px solid ${C.border}`, borderTop:`2px solid ${accent}`,
      borderRadius:6, padding:'14px 16px', ...style,
    }}>
      <div style={{ display:'flex', justifyContent:'space-between',
                    alignItems:'center', marginBottom:12 }}>
        <span style={{ color:accent, fontSize:10, fontWeight:700, letterSpacing:3 }}>{title}</span>
        {badge && <span style={{ color:C.dim, fontSize:9, border:`1px solid ${C.border}`,
          borderRadius:3, padding:'2px 6px' }}>{badge}</span>}
      </div>
      {children}
    </div>
  );
}

function M({ label, value, color = C.text, size = 20, sub }) {
  return (
    <div style={{ marginBottom:6 }}>
      <div style={{ color:C.dim, fontSize:9, letterSpacing:2 }}>{label}</div>
      <div style={{ color, fontSize:size, fontWeight:700,
                    fontFamily:'monospace', lineHeight:1.1 }}>{value}</div>
      {sub && <div style={{ color:C.dim, fontSize:9 }}>{sub}</div>}
    </div>
  );
}

function Dot({ ok }) {
  return <span style={{
    display:'inline-block', width:7, height:7, borderRadius:'50%',
    background: ok ? C.green : C.red,
    boxShadow:  ok ? `0 0 6px ${C.green}` : `0 0 6px ${C.red}`,
    marginRight:6, flexShrink:0,
  }}/>;
}

// ── Panel 1: P&L ─────────────────────────────────────────────────────────────
function PnlPanel({ data, live, dailyPnl }) {
  const last = data[data.length-1]?.pnl ?? 0;
  const peak = Math.max(...data.map(d => d.pnl), 0);
  const dd   = peak > 0 ? (peak - last) / peak * 100 : 0;
  const clr  = last >= 0 ? C.green : C.red;
  const dClr = dailyPnl >= 0 ? C.green : C.red;
  return (
    <Panel title="REALISED P&L (USDT)" badge={live ? 'LIVE' : 'CACHED'} accent={clr}>
      <div style={{ display:'flex', gap:20, marginBottom:8 }}>
        <M label="CUMULATIVE" value={`${fmt2(last)} USDT`} color={clr} />
        <M label="TODAY"      value={`${fmt2(dailyPnl)} USDT`} color={dClr} />
        <M label="DRAWDOWN"   value={fmtP(dd)}             color={dd > 3 ? C.red : C.sub} />
        <M label="PEAK"       value={`+${peak.toFixed(2)}`} color={C.sub} />
      </div>
      <ResponsiveContainer width="100%" height={110}>
        <AreaChart data={data} margin={{top:4,right:4,left:0,bottom:0}}>
          <defs>
            <linearGradient id="gPnl" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%"  stopColor={clr} stopOpacity={0.25}/>
              <stop offset="95%" stopColor={clr} stopOpacity={0}/>
            </linearGradient>
          </defs>
          <CartesianGrid stroke={C.border} strokeDasharray="3 3"/>
          <XAxis dataKey="t" hide/>
          <YAxis tick={{fill:C.dim,fontSize:9}} width={44}/>
          <Tooltip {...TT} formatter={v => [`${v.toFixed(2)} USDT`, 'P&L']}/>
          <ReferenceLine y={0} stroke={C.border}/>
          <Area type="monotone" dataKey="pnl" stroke={clr}
                fill="url(#gPnl)" dot={false} strokeWidth={2}/>
        </AreaChart>
      </ResponsiveContainer>
    </Panel>
  );
}

// ── Panel 2: Bid/Ask ──────────────────────────────────────────────────────────
function SpreadPanel({ data, live }) {
  const last   = data[data.length-1] ?? {};
  const spread = last.ask && last.bid ? (last.ask - last.bid).toFixed(2) : '—';
  const spClr  = parseFloat(spread) <= 0.20 ? C.green
               : parseFloat(spread) <= 0.50 ? C.amber : C.red;
  return (
    <Panel title="BID / ASK SPREAD" badge={live ? 'LIVE OB' : 'CACHED'} accent={C.teal}>
      <div style={{ display:'flex', gap:20, marginBottom:8 }}>
        <M label="BID"    value={last.bid?.toFixed(1) ?? '—'} color={C.green} />
        <M label="ASK"    value={last.ask?.toFixed(1) ?? '—'} color={C.red}   />
        <M label="SPREAD" value={`${spread} USDT`}            color={spClr}   />
      </div>
      <ResponsiveContainer width="100%" height={110}>
        <LineChart data={data} margin={{top:4,right:4,left:0,bottom:0}}>
          <CartesianGrid stroke={C.border} strokeDasharray="3 3"/>
          <XAxis dataKey="t" hide/>
          <YAxis tick={{fill:C.dim,fontSize:9}} width={52} domain={['auto','auto']}/>
          <Tooltip {...TT}/>
          <Line type="monotone" dataKey="bid" stroke={C.green} dot={false} strokeWidth={1.5}/>
          <Line type="monotone" dataKey="ask" stroke={C.red}   dot={false} strokeWidth={1.5}/>
        </LineChart>
      </ResponsiveContainer>
    </Panel>
  );
}

// ── Panel 3: Signal / IC proxy ────────────────────────────────────────────────
function ICPanel({ data, live }) {
  const last = data[data.length-1] ?? {};
  const sig  = last.signal ?? 0;
  const comp = last.composite ?? 0;
  const clr  = Math.abs(sig) > 0.02 ? C.green : Math.abs(sig) > 0 ? C.amber : C.red;
  return (
    <Panel title="COMPOSITE SIGNAL / OBI" badge={live ? 'LIVE' : 'CACHED'} accent={C.purple}>
      <div style={{ display:'flex', gap:20, marginBottom:8 }}>
        <M label="SIGNAL"    value={sig.toFixed(4)}   color={clr}   />
        <M label="COMPOSITE" value={comp.toFixed(4)}  color={C.sub} />
        <M label="OBI-5"     value={(last.obi5 ?? 0).toFixed(4)} color={C.sub} />
      </div>
      <ResponsiveContainer width="100%" height={110}>
        <LineChart data={data} margin={{top:4,right:4,left:0,bottom:0}}>
          <CartesianGrid stroke={C.border} strokeDasharray="3 3"/>
          <XAxis dataKey="t" hide/>
          <YAxis tick={{fill:C.dim,fontSize:9}} width={40} domain={[-1, 1]}/>
          <ReferenceLine y={0} stroke={C.amber} strokeDasharray="4 4"/>
          <Tooltip {...TT}/>
          <Line type="monotone" dataKey="signal"    stroke={C.purple} dot={false} strokeWidth={2}   name="Signal"/>
          <Line type="monotone" dataKey="composite" stroke={C.blue}   dot={false} strokeWidth={1.5} name="Composite"/>
        </LineChart>
      </ResponsiveContainer>
    </Panel>
  );
}

// ── Panel 4: Inventory ────────────────────────────────────────────────────────
function InventoryPanel({ inventory, regime, live }) {
  const MAX  = 0.10;   // Run-15 validated MAX_INVENTORY_BTC
  const pct  = clamp(Math.abs(inventory) / MAX * 100, 0, 100);
  const clr  = pct > 80 ? C.red : pct > 60 ? C.amber : C.green;
  const side = inventory > 0 ? 'LONG' : inventory < 0 ? 'SHORT' : 'FLAT';
  const regClr = regime === 'high_vol' ? C.amber
               : regime === 'halted'   ? C.red : C.green;
  return (
    <Panel title="INVENTORY POSITION" badge={live ? 'LIVE' : 'CACHED'} accent={clr}>
      <div style={{ display:'flex', gap:20, marginBottom:10 }}>
        <M label="POSITION (BTC)" value={inventory.toFixed(6)} color={clr} />
        <M label="SIDE"           value={side}                 color={clr} />
        <M label="UTILISATION"    value={fmtP(pct)}            color={clr} />
      </div>
      <div style={{ background:C.border, borderRadius:3, height:10, overflow:'hidden' }}>
        <div style={{
          width:`${pct}%`, height:'100%', borderRadius:3,
          background:`linear-gradient(90deg,${C.green},${clr})`,
          transition:'width 0.4s ease',
        }}/>
      </div>
      <div style={{ display:'flex', justifyContent:'space-between', marginTop:4 }}>
        <span style={{ color:C.dim, fontSize:9 }}>0</span>
        <span style={{ color:C.dim, fontSize:9 }}>MAX {MAX} BTC</span>
      </div>
      <div style={{ marginTop:8 }}>
        <M label="REGIME" value={(regime || 'unknown').toUpperCase()}
           color={regClr} size={12} />
      </div>
    </Panel>
  );
}

// ── Panel 5: Fill Rate ────────────────────────────────────────────────────────
function FillPanel({ data, live }) {
  const last    = data[data.length-1] ?? {};
  const fillPct = (last.fill_rate ?? 0) * 100;
  const fills   = last.total_fills ?? 0;
  const quotes  = last.total_quotes ?? 0;
  const fillClr = fillPct >= 5 ? C.green : fillPct >= 2 ? C.amber : C.red;
  return (
    <Panel title="FILL RATE & EXECUTION" badge={live ? 'LIVE' : 'CACHED'} accent={C.amber}>
      <div style={{ display:'flex', gap:20, marginBottom:8 }}>
        <M label="FILL RATE"    value={fmtP(fillPct)}  color={fillClr} />
        <M label="FILLS/QUOTES" value={`${fills}/${quotes}`} color={C.sub} sub="gate ≥ 5%" />
        <M label="GATE" value={fillPct >= 5 ? '✓ PASS' : '✗ FAIL'}
           color={fillClr} size={14} />
      </div>
      <ResponsiveContainer width="100%" height={110}>
        <BarChart data={data.slice(-20)} margin={{top:4,right:4,left:0,bottom:0}}>
          <CartesianGrid stroke={C.border} strokeDasharray="3 3"/>
          <XAxis dataKey="t" hide/>
          <YAxis tick={{fill:C.dim,fontSize:9}} width={30} domain={[0,30]}/>
          <ReferenceLine y={5} stroke={C.green} strokeDasharray="4 4"/>
          <Tooltip {...TT} formatter={v => [`${v.toFixed(2)}%`, 'Fill Rate']}/>
          <Bar dataKey="fill_rate_pct" fill={C.blue} radius={[2,2,0,0]}/>
        </BarChart>
      </ResponsiveContainer>
    </Panel>
  );
}

// ── Panel 6: Health ───────────────────────────────────────────────────────────
function HealthPanel({ services, tickBuf, dataAge, alerts, live }) {
  const svcs = [
    { name:'Kafka',       ok: services?.kafka       ?? false },
    { name:'Redis',       ok: services?.redis        ?? false },
    { name:'TimescaleDB', ok: services?.timescaledb  ?? false },
    { name:'Python Eng',  ok: services?.python_eng   ?? false },
    { name:'Claude Agent',ok: services?.claude_agent ?? false },
    { name:'Ray Head',    ok: services?.ray_head     ?? false },
  ];
  const last = alerts[alerts.length-1];
  const aClr = last?.severity === 'CRITICAL' ? C.red
             : last?.severity === 'WARNING'  ? C.amber : C.sub;
  const ageClr = dataAge < 2 ? C.green : dataAge < 10 ? C.amber : C.red;

  return (
    <Panel title="SERVICE HEALTH" badge={live ? 'LIVE' : 'OFFLINE'}
           accent={live ? C.green : C.red}>
      <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr',
                    gap:'5px 12px', marginBottom:10 }}>
        {svcs.map(s => (
          <div key={s.name} style={{ display:'flex', alignItems:'center', fontSize:11 }}>
            <Dot ok={s.ok}/>
            <span style={{ color: s.ok ? C.text : C.red }}>{s.name}</span>
          </div>
        ))}
      </div>
      <div style={{ display:'flex', gap:16, marginBottom:8 }}>
        <M label="TICK BUFFER" value={tickBuf ?? '—'}  color={C.sub} size={13}/>
        <M label="DATA AGE"    value={dataAge != null ? `${dataAge}s` : '—'}
           color={ageClr} size={13}/>
      </div>
      {last && (
        <div style={{
          background:C.bg, border:`1px solid ${C.border}`,
          borderLeft:`3px solid ${aClr}`, borderRadius:3,
          padding:'6px 10px', fontSize:9, color:aClr, lineHeight:1.5,
        }}>
          <span style={{ color:C.dim }}>[{last.severity}] </span>
          {(last.message || last.category || '').slice(0, 120)}
        </div>
      )}
    </Panel>
  );
}

// =============================================================================
// MAIN APP — polls /metrics every 2s
// =============================================================================
export default function App() {
  const [live,      setLive]      = useState(false);
  const [tick,      setTick]      = useState(0);
  const [lastPnl,   setLastPnl]   = useState(0);
  const [dailyPnl,  setDailyPnl]  = useState(0);
  const [pnlData,   setPnlData]   = useState([{ t:0, pnl:0 }]);
  const [obData,    setObData]    = useState([]);
  const [sigData,   setSigData]   = useState([{ t:0, signal:0, composite:0, obi5:0 }]);
  const [fillData,  setFillData]  = useState([{ t:0, fill_rate_pct:0, total_fills:0, total_quotes:0 }]);
  const [inventory, setInventory] = useState(0);
  const [regime,    setRegime]    = useState('unknown');
  const [services,  setServices]  = useState({});
  const [tickBuf,   setTickBuf]   = useState(null);
  const [dataAge,   setDataAge]   = useState(null);
  const [alerts,    setAlerts]    = useState([]);
  const [error,     setError]     = useState(null);

  // ── Poll /metrics every 2s ─────────────────────────────────────────────────
  useEffect(() => {
    const poll = async () => {
      try {
        const res = await fetch(METRICS_URL);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const d = await res.json();
        const now = Date.now();

        setLive(true);
        setError(null);
        setTick(t => t + 1);

        // P&L: read directly from agent metrics (sourced from Redis)
        const cumPnl   = d.cumulative_pnl ?? d.pnl ?? 0;
        const dayPnl   = d.daily_pnl ?? 0;
        setLastPnl(cumPnl);
        setDailyPnl(dayPnl);
        setPnlData(a => push(a, { t: now, pnl: cumPnl }));

        // OB data
        if (d.best_bid && d.best_ask)
          setObData(a => push(a, { t:now, bid:d.best_bid, ask:d.best_ask }));

        // Signal data
        setSigData(a => push(a, {
          t:         now,
          signal:    d.signal    ?? 0,
          composite: d.composite ?? 0,
          obi5:      d.obi5      ?? 0,
        }));

        // Fill rate
        setFillData(a => push(a, {
          t:             now,
          fill_rate_pct: (d.fill_rate ?? 0) * 100,
          total_fills:   d.total_fills  ?? 0,
          total_quotes:  d.total_quotes ?? 0,
        }));

        // Inventory & regime
        setInventory(d.inventory ?? 0);
        setRegime(d.regime ?? 'unknown');

        // Services & health
        setServices(d.services ?? {});
        setTickBuf(d.tick_buffer ?? null);
        setDataAge(d.data_age_sec ?? null);

      } catch (e) {
        setLive(false);
        setError(e.message);
      }
    };

    poll();
    const id = setInterval(poll, POLL_MS);
    return () => clearInterval(id);
  }, []);

  // ── Poll /alerts/recent every 10s ─────────────────────────────────────────
  useEffect(() => {
    const pollAlerts = async () => {
      try {
        const res = await fetch(ALERTS_URL);
        const rows = await res.json();
        if (Array.isArray(rows) && rows.length)
          setAlerts(rows.slice(0, 20));
      } catch(_) {}
    };
    pollAlerts();
    const id = setInterval(pollAlerts, ALERT_MS);
    return () => clearInterval(id);
  }, []);

  const pnlClr  = lastPnl  >= 0 ? C.green : C.red;
  const dPnlClr = dailyPnl >= 0 ? C.green : C.red;

  return (
    <div style={{ background:C.bg, minHeight:'100vh', padding:'16px 20px',
      fontFamily:'"Courier New","Lucida Console",monospace' }}>

      <style>{`
        @keyframes pulse{0%,100%{opacity:1}50%{opacity:.4}}
        *{box-sizing:border-box}
        ::-webkit-scrollbar{width:4px}
        ::-webkit-scrollbar-track{background:${C.bg}}
        ::-webkit-scrollbar-thumb{background:${C.border};border-radius:2px}
      `}</style>

      {/* Header */}
      <div style={{ display:'flex', justifyContent:'space-between',
        alignItems:'flex-start', marginBottom:16,
        borderBottom:`1px solid ${C.border}`, paddingBottom:12 }}>
        <div>
          <div style={{ fontSize:24, fontWeight:900, letterSpacing:6, color:C.green,
            textShadow:`0 0 20px ${C.green}44` }}>HFT MARKET MAKING</div>
          <div style={{ color:C.dim, fontSize:10, letterSpacing:3, marginTop:2 }}>
            BTCUSDT · BYBIT SPOT · AVELLANEDA-STOIKOV · PAPER TRADING
          </div>
        </div>
        <div style={{ textAlign:'right' }}>
          <div style={{ display:'flex', alignItems:'center',
            justifyContent:'flex-end', gap:6, marginBottom:4 }}>
            <span style={{
              display:'inline-block', width:7, height:7, borderRadius:'50%',
              background: live ? C.green : C.red,
              boxShadow:  live ? `0 0 8px ${C.green}` : `0 0 8px ${C.red}`,
              animation:  live ? 'pulse 2s ease-in-out infinite' : 'none',
            }}/>
            <span style={{ fontSize:11, fontWeight:700, letterSpacing:2,
              color: live ? C.green : C.red }}>
              {live ? 'LIVE' : 'DISCONNECTED'}
            </span>
          </div>
          <div style={{ color:C.dim, fontSize:10 }}>
            TICK #{tick} · {new Date().toLocaleTimeString()}
          </div>
          <div style={{ color:dPnlClr, fontSize:18, fontWeight:700 }}>
            {fmt2(dailyPnl)} USDT <span style={{ fontSize:11, color:C.dim }}>TODAY</span>
          </div>
          <div style={{ color:pnlClr, fontSize:12, fontWeight:700, opacity:0.7 }}>
            {fmt2(lastPnl)} USDT <span style={{ fontSize:10, color:C.dim }}>CUMULATIVE</span>
          </div>
        </div>
      </div>

      {/* Error / disconnected banner */}
      {!live && (
        <div style={{ background:`${C.amber}18`, border:`1px solid ${C.amber}55`,
          borderRadius:4, padding:'8px 14px', marginBottom:12,
          fontSize:11, color:C.amber, letterSpacing:1 }}>
          ⚠ Cannot reach {METRICS_URL} — {error ?? 'retrying...'}
          <br/>Ensure sitaram-claude-agent is running and CORS is enabled.
        </div>
      )}

      {/* 6-panel grid */}
      <div style={{ display:'grid', gridTemplateColumns:'repeat(3,1fr)', gap:14 }}>
        <PnlPanel     data={pnlData}  live={live} dailyPnl={dailyPnl}/>
        <SpreadPanel  data={obData}   live={live}/>
        <ICPanel      data={sigData}  live={live}/>
        <InventoryPanel inventory={inventory} regime={regime} live={live}/>
        <FillPanel    data={fillData} live={live}/>
        <HealthPanel  services={services} tickBuf={tickBuf}
                      dataAge={dataAge} alerts={alerts} live={live}/>
      </div>

      {/* Footer */}
      <div style={{ textAlign:'center', marginTop:14, color:C.dim, fontSize:9,
        letterSpacing:3, borderTop:`1px solid ${C.border}`, paddingTop:10 }}>
        SITARAM HFT v3.0 · PAPER TRADING · AS MODEL RUN-15 ·
        AGENT {live ? 'CONNECTED' : 'OFFLINE'} · {new Date().toLocaleDateString()}
      </div>
    </div>
  );
}
