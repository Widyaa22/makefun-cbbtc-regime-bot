// cbBTC trend-regime bot — agent-fun/bars-replay/v1, long only, one symbol (CBBTC).
//
// What this market is: a 1-minute replay of cbBTC/USDC on one Orca Whirlpool. Most minutes are
// flat, the drift is slow (mean absolute 1-minute move ~4 bps, sd ~7 bps over the dev window), and
// a full round trip costs ~26-32 bps: 4 bps fee each way, 2 bps modelled slippage each way, and
// ~10 bps of price impact each way at full size against the pool's ~1M USDC depth. Buy-and-hold
// loses on a falling day (the dev window fell 2.8%: buy-and-hold -31.6 USDC), a fast EMA cross
// bleeds the fee 80 times (-148 USDC on dev), and cash earns nothing. The only edge in a tape like
// this is to ride one multi-hour up-leg per day from a cheap entry and sit in cash otherwise.
//
// The machine, in the order it runs on each close:
//   FOLD     — push the close, step a 60-minute EMA, read its change over the last 40 bars in bps.
//   REGIME   — long only while that slope is >= +5 bps (~1.8%/day of drift). Below it the drift
//              cannot pay a round trip, so cash is the correct position.
//   TRIGGER  — inside that uptrend, buy a 30-minute pullback of at least 30 bps. Dips are the cheap
//              entries: measured on dev, demanding the close be *above* the EMA first cut the result
//              from +12.5 to +2.5 USDC, because the pullback itself pays the fee and the impact.
//   EXIT     — sell the whole position on a confirmed trend break (slope negative 10 closes in a
//              row) or a 120 bps trailing stop under the peak close.
//   GUARDS   — 150 bps hard stop from the entry close, a 10-bar minimum hold before the trailing
//              stop may fire, and a 30-bar cooldown after any exit so a stopped-out position is not
//              re-bought into the same falling tape.
//   SKIP     — a 60 bps profit give-back exit was tried and rejected: it cut the legs short and
//              turned +12.5 into +4.0 USDC. The trend run itself is the edge; do not clip it.
//
// Every number is an integer: prices are the harness's quote-scaled BigInt integers and every ratio
// is bps with BigInt division (truncating toward zero), so two graded executions emit identical
// orders. No clock, no randomness, no I/O.

import { BPS, runBars, type Bar, type BarsState, type BuyInput, type OrderInput, type SellInput } from './vendor/makefun-bars.ts';

// ---------------------------------------------------------------- parameters (bars, bps)
const SLOW_BARS = 60;        // the regime line: an hour of 1-minute closes
const SLOPE_LAG = 40;        // the regime test: the EMA's change over 40 bars
const SLOPE_MIN_BPS = 5n;    // the trend must drift >= ~1.8%/day before a buy is worth its cost
const DIP_LAG = 30;          // the pullback window, minutes
const DIP_BPS = 30n;         // ... deep enough to pay the fee and the impact
const TRAIL_BPS = 120n;      // trailing stop, ~4x the round-trip cost, under the peak close
const EXIT_NEG_BARS = 10;    // trend break: the EMA slope negative this many closes in a row
const HARD_STOP_BPS = 150n;  // from the entry close: a crash guard, never the normal exit
const MIN_HOLD_BARS = 10;    // the trailing stop waits this long after the entry
const COOLDOWN_BARS = 30;    // no re-entry for this long after any exit
const SIZE_BPS = 9800n;      // 98% of equity per buy: fee and impact stay inside cash
const MAX_SLIPPAGE_BPS = 100; // refuse a fill more than 1% off the close it was decided on

// ---------------------------------------------------------------- per-symbol state
interface Sym {
  closes: bigint[];   // every close, oldest first
  emas: bigint[];     // the regime EMA after each close, same length
  slope: bigint | null; // the EMA's change over SLOPE_LAG bars, in bps, after the latest close
  peak: bigint;       // highest close held: the trailing stop's reference
  negRun: number;     // consecutive closes with a negative EMA slope
  entry: bigint;      // the close the position was entered at
  entryBar: number;   // the bar number of that close
  cool: number;       // bars left before the next entry is allowed
  bar: number;        // closes folded so far
}

const book = new Map<string, Sym>();

function stateOf(sym: string): Sym {
  let s = book.get(sym);
  if (s === undefined) {
    s = { closes: [], emas: [], slope: null, peak: 0n, negRun: 0, entry: 0n, entryBar: -1, cool: 0, bar: 0 };
    book.set(sym, s);
  }
  return s;
}

/** How far `close` has fallen under `from`, in bps. */
function fallBps(from: bigint, close: bigint): bigint {
  return ((from - close) * BPS) / from;
}

/**
 * Fold one close in. The EMA is seeded with the first close and then moves 2/(span+1) of the way to
 * each new one; the slope is read straight after, so every branch of the strategy sees the same
 * regime number for this close.
 */
function fold(sym: string, close: bigint): Sym {
  const s = stateOf(sym);
  const prev = s.emas.length === 0 ? close : s.emas[s.emas.length - 1]!;
  s.emas.push(prev + (2n * (close - prev)) / BigInt(SLOW_BARS + 1));
  s.closes.push(close);
  s.bar++;
  if (s.cool > 0) s.cool--;
  const n = s.emas.length;
  s.slope = n <= SLOPE_LAG ? null : ((s.emas[n - 1]! - s.emas[n - 1 - SLOPE_LAG]!) * BPS) / s.emas[n - 1 - SLOPE_LAG]!;
  if (s.slope !== null) s.negRun = s.slope < 0n ? s.negRun + 1 : 0;
  return s;
}

/** The close's change over `lag` bars in bps of the older close, or null before there is history. */
function changeBps(s: Sym, lag: number): bigint | null {
  const n = s.closes.length;
  if (n <= lag) return null;
  const then = s.closes[n - 1 - lag]!;
  return ((s.closes[n - 1]! - then) * BPS) / then;
}

/** Sell the whole position? Trend break every close; the price stops only after MIN_HOLD_BARS. */
function wantsExit(s: Sym, close: bigint): boolean {
  if (s.negRun >= EXIT_NEG_BARS) return true;
  if (close > s.peak) s.peak = close;
  if (s.bar - s.entryBar < MIN_HOLD_BARS) return false;
  if (fallBps(s.peak, close) >= TRAIL_BPS) return true;
  return fallBps(s.entry, close) >= HARD_STOP_BPS;
}

/** The size of the buy this close justifies, or 0 for none. */
function entryNotional(s: Sym, state: BarsState, symbol: string): bigint {
  if (s.cool > 0) return 0n;
  const dip = changeBps(s, DIP_LAG);
  if (s.slope === null || dip === null) return 0n;
  if (s.slope < SLOPE_MIN_BPS || dip > -DIP_BPS) return 0n;
  // 98% of equity, never more than the cash on hand or the weight cap's room, and never under the
  // market's minimum notional: the whole fee and the modelled impact stay inside the cash.
  const byEquity = (state.equity * SIZE_BPS) / BPS;
  const room = state.weightRoom(symbol);
  const cash = state.cash;
  let notional = byEquity < cash ? byEquity : cash;
  if (notional > room) notional = room;
  return notional < BigInt(state.hello.fees.min_notional_quote_units) ? 0n : notional;
}

// ---------------------------------------------------------------- the loop
await runBars({
  onBatch(state, batch) {
    // Fold every bar of the batch first: indicators must see a warm-up bar's close too, so the
    // first tradable batch already has a regime to read.
    const fresh = new Set<string>();
    for (const bar of batch.bars as Bar[]) {
      fold(bar.symbol, BigInt(bar.c));
      fresh.add(bar.symbol);
    }
    // Warm-up batches are not tradable: the replay rejects their orders (`warmup`).
    if (state.warmingUp) return [];

    const orders: OrderInput[] = [];
    const { caps } = state.hello;
    const holding = state.positions.size > 0;

    for (const { symbol } of state.hello.symbols) {
      if (!fresh.has(symbol)) continue;
      const s = stateOf(symbol);
      if (s.slope === null) continue;
      const close = s.closes[s.closes.length - 1]!;

      const position = state.positions.get(symbol);
      if (position !== undefined && position.qty > 0n) {
        if (wantsExit(s, close)) {
          orders.push({ side: 'sell', symbol, qty: position.qty, max_slippage_bps: MAX_SLIPPAGE_BPS } satisfies SellInput);
          s.cool = COOLDOWN_BARS;
        }
        continue;
      }

      // Flat. One position at a time across the universe; a reply carries at most the order cap.
      if (holding || orders.length >= caps.max_orders_per_bar) continue;
      const notional = entryNotional(s, state, symbol);
      if (notional === 0n) continue;
      orders.push({ side: 'buy', symbol, notional_quote_units: notional, max_slippage_bps: MAX_SLIPPAGE_BPS } satisfies BuyInput);
      s.peak = close;
      s.entry = close;
      s.entryBar = s.bar;
      s.negRun = 0;
      break;
    }
    return orders;
  },
});