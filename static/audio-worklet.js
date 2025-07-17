class AudioProcessor extends AudioWorkletProcessor {
    constructor() {
        super();
        this.bufferSize = 4096;
        this.buffer = new Float32Array(this.bufferSize);
        this.bufferIndex = 0;
    }
    
    process(inputs, outputs, parameters) {
        const input = inputs[0];
        
        if (input.length > 0) {
            const inputData = input[0];
            
            for (let i = 0; i < inputData.length; i++) {
                this.buffer[this.bufferIndex] = inputData[i];
                this.bufferIndex++;
                
                if (this.bufferIndex >= this.bufferSize) {
                    // Calculate audio level
                    let sum = 0;
                    for (let j = 0; j < this.bufferSize; j++) {
                        sum += this.buffer[j] * this.buffer[j];
                    }
                    const level = Math.sqrt(sum / this.bufferSize);
                    
                    // Send audio data to main thread
                    this.port.postMessage({
                        audioData: Array.from(this.buffer),
                        level: level
                    });
                    
                    this.bufferIndex = 0;
                }
            }
        }
        
        return true;
    }
}

registerProcessor('audio-processor', AudioProcessor);
