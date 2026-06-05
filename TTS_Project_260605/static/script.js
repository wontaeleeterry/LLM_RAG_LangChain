const askBtn =
    document.getElementById("askBtn");

const clearBtn =
    document.getElementById("clearBtn");

const inputText =
    document.getElementById("inputText");

const responseBox =
    document.getElementById("responseBox");

const voiceSelect =
    document.getElementById("voiceSelect");

const spinner =
    document.getElementById("spinner");

const player =
    document.getElementById("player");


askBtn.addEventListener(
    "click",
    async () => {

        const text =
            inputText.value.trim();

        if(!text){

            alert(
                "Please enter text."
            );

            return;
        }

        spinner.style.display =
            "block";

        askBtn.disabled = true;

        try{

            // STEP 1
            // Ask LLM

            const llmResponse =
                await fetch(
                    "/ask",
                    {
                        method:"POST",

                        headers:{
                            "Content-Type":
                            "application/json"
                        },

                        body:JSON.stringify(
                            {
                                text:text
                            }
                        )
                    }
                );

            const llmData =
                await llmResponse.json();

            const answer =
                llmData.response;

            responseBox.textContent =
                answer;

            // STEP 2
            // Generate TTS

            const audioResponse =
                await fetch(
                    "/speak",
                    {
                        method:"POST",

                        headers:{
                            "Content-Type":
                            "application/json"
                        },

                        body:JSON.stringify(
                            {
                                text:answer,
                                voice:
                                voiceSelect.value
                            }
                        )
                    }
                );

            const blob =
                await audioResponse.blob();

            const audioURL =
                URL.createObjectURL(blob);

            player.src =
                audioURL;

            await player.play();

        }
        catch(error){

            console.error(error);

            responseBox.textContent =
                "Error occurred.";

        }
        finally{

            spinner.style.display =
                "none";

            askBtn.disabled =
                false;
        }
    }
);


clearBtn.addEventListener(
    "click",
    () => {

        inputText.value = "";

        responseBox.textContent =
            "";

        player.pause();

        player.src = "";
    }
);