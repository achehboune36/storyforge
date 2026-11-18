// Reconnecting WebSocket client.
//
// Ported near-verbatim from gameshow/frontend/ws.js, which has two ideas worth
// keeping: dispatch by handler-map lookup rather than a switch (so adding a
// server message type never breaks an old client), and close code 4001 as the
// sentinel meaning "auth failed, stop retrying".
//
// Added here: an app-level heartbeat. gameshow relied purely on the protocol
// ping, which is fine for short question rounds but not for a socket that can
// sit quiet for 30s while an image generates behind a proxy.

export class StoryWS {
  constructor(sessionId, token, handlers) {
    this.sessionId = sessionId;
    this.token = token || '';
    this.handlers = handlers;
    this.ws = null;
    this.delay = 1000;
    this.maxDelay = 30000;
    this.shouldReconnect = true;
    this._timer = null;
    this._ping = null;
  }

  connect() {
    const proto = location.protocol === 'https:' ? 'wss' : 'ws';
    const url = `${proto}://${location.host}/ws/session/${this.sessionId}`
      + (this.token ? `?token=${encodeURIComponent(this.token)}` : '');
    this.ws = new WebSocket(url);

    this.ws.onopen = () => {
      this.delay = 1000;
      this._ping = setInterval(() => this.send('ping', {}), 20000);
      this.handlers.onOpen?.();
    };

    this.ws.onmessage = (e) => {
      let msg;
      try { msg = JSON.parse(e.data); } catch { return; }
      this.handlers[msg.t]?.(msg.d || {});
    };

    this.ws.onclose = (e) => {
      clearInterval(this._ping);
      if (e.code === 4001) {
        this.shouldReconnect = false;
        this.handlers.onRejected?.(e.reason);
        return;
      }
      this.handlers.onClosed?.();
      if (this.shouldReconnect) {
        this._timer = setTimeout(() => this.connect(), this.delay);
        this.delay = Math.min(this.delay * 2, this.maxDelay);
      }
    };

    this.ws.onerror = () => {};
  }

  send(t, d) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ t, d }));
      return true;
    }
    return false;
  }

  close() {
    this.shouldReconnect = false;
    clearTimeout(this._timer);
    clearInterval(this._ping);
    this.ws?.close();
  }
}
