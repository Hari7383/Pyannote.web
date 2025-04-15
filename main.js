mediaRecorder.onstop = async () => {
    const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
    const formData = new FormData();
    formData.append('audio', audioBlob, 'recording.wav');

    try {
        const response = await fetch('/process-audio', {
            method: 'POST',
            body: formData
        });

        const result = await response.json();
        console.log(result); // { speakers: { SPEAKER_00: "text", SPEAKER_01: "text" } }

        // Show transcription result
        let resultText = '';
        for (const [speaker, text] of Object.entries(result.speakers)) {
            resultText += `<p><strong>${speaker}</strong>: ${text}</p>`;
        }

        document.body.insertAdjacentHTML('beforeend', `<div style="color:white;position:absolute;bottom:5%;text-align:center;">${resultText}</div>`);
    } catch (err) {
        console.error('Transcription error:', err);
    }
};
