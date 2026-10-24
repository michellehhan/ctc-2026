#!/usr/bin/env python3
"""REST client for the exchange, shared by bot.py and manual.py.

    from exchange import Exchange, load_key
    ex = Exchange(load_key())
    ex.place_order("PORT", "buy", 10, 71.2)      # limit; price=None -> market
    ex.cancel(order_id); ex.cancel_all("PORT"); ex.cancel_all()
    ex.open_orders(); ex.positions(); ex.orderbook("PORT"); ex.time(); ex.case()
    ex.etf_create("ARCH", 2); ex.etf_redeem("REBLD", 1)

Every call returns the decoded JSON (or raises ExchangeError with the status and body). The response
formats of /positions and /orders/open are not documented in the case packet, so `parse_positions` and
`parse_open_orders` accept the common shapes (list of objects or dict keyed by symbol) and log the raw
payload the first time so the format can be checked on the testnet.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from typing import Any

import requests

BASE_URL = "https://cornelltradingcompetition.org"
USER_AGENT = "ctc-hurricane-bot/1.0"  # the network refuses the default python User-Agent

log = logging.getLogger("exchange")


class ExchangeError(Exception):
    def __init__(self, status: int, body: str, what: str = ""):
        super().__init__(f"{what} -> HTTP {status}: {body[:300]}")
        self.status, self.body = status, body


def load_key() -> str:
    """CTC_API_KEY from the environment; exit with a clear message when it is missing."""
    key = os.environ.get("CTC_API_KEY", "").strip()
    if not key:
        sys.exit("CTC_API_KEY is not set. Run:  export CTC_API_KEY=your_bot_key   (from the Team page), then start again.")
    return key


class Exchange:
    def __init__(self, key: str, base_url: str = BASE_URL, timeout: float = 5.0):
        self.base = base_url.rstrip("/")
        self.timeout = timeout
        self.s = requests.Session()
        self.s.headers.update({"X-API-Key": key, "User-Agent": USER_AGENT, "Accept": "application/json"})
        self._logged_shapes: set[str] = set()

    # ------------------------------------------------------------------ raw calls
    def _req(self, method: str, path: str, body: dict | None = None, api: bool = True) -> Any:
        url = self.base + ("/api/v1" if api else "") + path
        try:
            r = self.s.request(method, url, json=body, timeout=self.timeout)
        except requests.RequestException as exc:
            raise ExchangeError(0, str(exc), f"{method} {path}") from exc
        if r.status_code >= 400:
            raise ExchangeError(r.status_code, r.text, f"{method} {path} {json.dumps(body) if body else ''}")
        if not r.content:
            return {}
        try:
            return r.json()
        except ValueError:
            return {"raw": r.text}

    def place_order(self, symbol: str, side: str, quantity: int, price: float | None = None) -> dict:
        body = {"symbol": symbol, "side": side, "order_type": "limit" if price is not None else "market", "quantity": int(quantity)}
        if price is not None:
            body["price"] = round(float(price), 2)
        return self._req("POST", "/orders", body)

    def cancel(self, order_id: str) -> Any:
        return self._req("DELETE", f"/orders/{order_id}")

    def cancel_all(self, symbol: str | None = None) -> Any:
        return self._req("DELETE", "/orders/all" + (f"?symbol={symbol}" if symbol else ""))

    def open_orders(self) -> Any:
        return self._req("GET", "/orders/open")

    def positions(self) -> Any:
        return self._req("GET", "/positions")

    def orderbook(self, symbol: str) -> Any:
        return self._req("GET", f"/orderbook/{symbol}")

    def time(self) -> Any:
        return self._req("GET", "/competitions/time")

    def case(self) -> Any:
        return self._req("GET", "/cases/hurricane.json", api=False)

    def etf_create(self, symbol: str, quantity: int) -> Any:
        return self._req("POST", f"/etf/{symbol}/create", {"quantity": int(quantity)})

    def etf_redeem(self, symbol: str, quantity: int) -> Any:
        return self._req("POST", f"/etf/{symbol}/redeem", {"quantity": int(quantity)})

    # ------------------------------------------------------------------ tolerant parsers
    def _log_shape(self, what: str, payload: Any) -> None:
        if what not in self._logged_shapes:
            self._logged_shapes.add(what)
            log.info("raw %s payload (first time): %s", what, json.dumps(payload)[:600])

    def parse_positions(self, payload: Any) -> dict[str, int]:
        """-> {symbol: signed quantity}. Accepts [{symbol, quantity|position|qty|net}], {symbol: qty}, {"positions": ...}."""
        self._log_shape("positions", payload)
        if isinstance(payload, dict):
            for key in ("positions", "data", "items", "results"):
                if key in payload and isinstance(payload[key], (list, dict)):
                    return self.parse_positions_inner(payload[key])
            return self.parse_positions_inner(payload)
        return self.parse_positions_inner(payload)

    @staticmethod
    def parse_positions_inner(payload: Any) -> dict[str, int]:
        out: dict[str, int] = {}
        if isinstance(payload, list):
            for item in payload:
                if not isinstance(item, dict):
                    continue
                sym = item.get("symbol") or item.get("ticker")
                qty = None
                for k in ("quantity", "position", "qty", "net", "net_position", "size"):
                    if k in item and isinstance(item[k], (int, float)):
                        qty = item[k]
                        break
                if sym and qty is not None:
                    out[sym] = int(round(qty))
        elif isinstance(payload, dict):
            for sym, v in payload.items():
                if isinstance(v, (int, float)):
                    out[sym] = int(round(v))
                elif isinstance(v, dict):
                    for k in ("quantity", "position", "qty", "net", "net_position", "size"):
                        if k in v and isinstance(v[k], (int, float)):
                            out[sym] = int(round(v[k]))
                            break
        return out

    def parse_open_orders(self, payload: Any) -> list[dict]:
        """-> [{id, symbol, side, quantity (remaining), price}]."""
        self._log_shape("open_orders", payload)
        items = payload
        if isinstance(payload, dict):
            for key in ("orders", "data", "items", "results", "open_orders"):
                if key in payload and isinstance(payload[key], list):
                    items = payload[key]
                    break
            else:
                items = list(payload.values()) if all(isinstance(v, dict) for v in payload.values()) else []
        out = []
        for o in items if isinstance(items, list) else []:
            if not isinstance(o, dict):
                continue
            oid = o.get("order_id") or o.get("id") or o.get("orderId")
            sym = o.get("symbol")
            side = str(o.get("side", "")).lower()
            rem = None
            for k in ("remaining_quantity", "remaining", "quantity_remaining", "open_quantity", "leaves_quantity"):
                if k in o and isinstance(o[k], (int, float)):
                    rem = o[k]
                    break
            if rem is None:
                q = o.get("quantity", 0) or 0
                f = o.get("filled_quantity", o.get("filled", 0)) or 0
                rem = q - f
            if oid is not None and sym:
                out.append({"id": str(oid), "symbol": sym, "side": "buy" if side.startswith("b") else "sell",
                            "quantity": int(round(rem)), "price": o.get("price")})
        return out

    @staticmethod
    def parse_order_response(resp: Any) -> tuple[str | None, str, int]:
        """-> (order_id, status, filled_quantity) from a POST /orders response."""
        if not isinstance(resp, dict):
            return None, "unknown", 0
        o = resp.get("order", resp) if isinstance(resp.get("order"), dict) else resp
        oid = o.get("order_id") or o.get("id") or o.get("orderId")
        status = str(o.get("status", "accepted")).lower()
        filled = 0
        for k in ("filled_quantity", "filled", "executed_quantity"):
            if k in o and isinstance(o[k], (int, float)):
                filled = int(o[k])
                break
        return (str(oid) if oid is not None else None), status, filled
