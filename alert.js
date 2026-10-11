(() => {
    const preferenceKey = "agriguard-alert-sound";
    const toggleButton = document.getElementById("toggleAlertSound");
    const testButton = document.getElementById("playAlertSound");
    const status = document.getElementById("soundStatus");
    if (!toggleButton || !testButton || !status) return;

    const alertAudio = new Audio("/static/alert.wav");
    alertAudio.preload = "auto";

    function soundEnabled() {
        return window.localStorage.getItem(preferenceKey) === "enabled";
    }

    function updateControls() {
        const enabled = soundEnabled();
        toggleButton.setAttribute("aria-pressed", String(enabled));
        toggleButton.textContent = enabled ? "🔊 Alert sound enabled" : "🔇 Enable alert sound";
        testButton.hidden = !enabled;
    }

    async function playAlertTone() {
        try {
            alertAudio.pause();
            alertAudio.currentTime = 0;
            await alertAudio.play();
            status.textContent = "Alert sound played.";
            return true;
        } catch (error) {
            status.textContent = "Alert sound could not play. Check your volume and use Play alert sound to try again.";
            testButton.hidden = false;
            return false;
        }
    }

    toggleButton.addEventListener("click", async () => {
        const enabled = soundEnabled();
        window.localStorage.setItem(preferenceKey, enabled ? "disabled" : "enabled");
        updateControls();
        if (enabled) {
            status.textContent = "Alert sound is off. Turn it on to hear high-risk alerts.";
            return;
        }
        await playAlertTone();
    });

    testButton.addEventListener("click", playAlertTone);
    updateControls();

    if (document.querySelector(".threat-alert") && soundEnabled()) {
        window.addEventListener("load", async () => {
            const played = await playAlertTone();
            if (!played) {
                status.textContent = "Tap Play alert sound to hear the alert tone.";
            }
        }, { once: true });
    }
})();
