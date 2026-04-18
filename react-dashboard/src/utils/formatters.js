export const fmt = {
  pnl:    v => (v >= 0 ? '+' : '') + v.toFixed(2),
  pct:    v => (v * 100).toFixed(1) + '%',
  price:  v => v.toFixed(1),
  btc:    v => v.toFixed(4),
  ic:     v => v.toFixed(4),
};
