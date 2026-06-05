const enterBtn =
    document.getElementById("enterBtn");

const clearBtn =
    document.getElementById("clearBtn");

const inputText =
    document.getElementById("inputText");

const player =
    document.getElementById("player");

const voiceSelect =
    document.getElementById("voiceSelect");

const loadingContainer =
    document.getElementById("loadingContainer");


enterBtn.addEventListener(
    "click",
    async () => {

        const text =
            inputText.value.trim();

        if (!text) {

            alert(
                "텍스트를 입력하세요."
            );

            return;
        }

        try {

            loadingContainer.style.display =
                "block";

            enterBtn.disabled = true;

            clearBtn.disabled = true;

            enterBtn.textContent =
                "Generating...";

            const formData =
                new FormData();

            formData.append(
                "text",
                text
            );

            formData.append(
                "voice",
                voiceSelect.value
            );

            const response =
                await fetch(
                    "/tts",
                    {
                        method:"POST",
                        body:formData
                    }
                );

            if (!response.ok) {

                throw new Error(
                    "TTS generation failed"
                );
            }

            const blob =
                await response.blob();

            const audioURL =
                URL.createObjectURL(blob);

            player.src =
                audioURL;

            await player.play();

        }
        catch(error){

            console.error(error);

            alert(
                "음성 생성 중 오류가 발생했습니다."
            );
        }
        finally{

            loadingContainer.style.display =
                "none";

            enterBtn.disabled = false;

            clearBtn.disabled = false;

            enterBtn.textContent =
                "Enter";
        }
    }
);


clearBtn.addEventListener(
    "click",
    () => {

        inputText.value = "";

        player.pause();

        player.src = "";
    }
);


// Ctrl + Enter 지원
inputText.addEventListener(
    "keydown",
    (event) => {

        if (
            event.ctrlKey &&
            event.key === "Enter"
        ) {

            enterBtn.click();
        }
    }
);