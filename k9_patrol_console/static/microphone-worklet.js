class Go2Microphone extends AudioWorkletProcessor {
  constructor() {
    super();
    this.buffer = new ArrayBuffer(3840);
    this.pcm = new DataView(this.buffer);
    this.count = 0;
  }
  process(inputs) {
    const input = inputs[0]?.[0];
    if (!input) return true;
    for (const sample of input) {
      this.pcm.setInt16(2 * this.count++, Math.round(Math.max(-1, Math.min(1, sample)) * 32767), true);
      if (this.count === 1920) {
        this.port.postMessage(this.buffer, [this.buffer]);
        this.buffer = new ArrayBuffer(3840);
        this.pcm = new DataView(this.buffer);
        this.count = 0;
      }
    }
    return true;
  }
}
registerProcessor('go2-microphone', Go2Microphone);
