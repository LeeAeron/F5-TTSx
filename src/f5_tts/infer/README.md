# 🚀 F5-TTSx: A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching eXtended
<img src="https://raw.githubusercontent.com/LeeAeron/F5TTSx/img/F5TTSx.png?raw=true" />

[![Release](https://img.shields.io/github/release/LeeAeron/F5-TTSx.svg)](https://github.com/LeeAeron/F5-TTSx/releases/latest)


## 🔧 About
**F5-TTSx** is a custom build of the F5-TTS voice synthezer, designed to get more fliud settings for synthezing voice - especially on systems with limited VRAM.

## 🔧 Key Differences from Official F5-TTSx

- Optimized for all systems with nVidia GPUs
- Main `.bat` menu with option to install/reinstall project
- Enhanced UI settings with useful features and stability improvements  

### ✅ Supported Models

- all official F5-TTS models, including fine-tuned customs (by suppport them with Custom profile)
- Qwen 2.5/3B Instruct and Microsoft Phi 4 mini instruct for Ai voice chat

### 🖥️ Windows Installation

This F5-TTSx will be provided with only *.bat installer/re-installer/starter file, that will download and install all components and build Portable F5-TTSx.


➤ Please Note:
    - I'm supporting only nVidia GTX10xx-16xx and RTX20xx-50xx GPUs.
    - This installer is intended for those running Windows 10 or higher. 
    - Application functionality for systems running Windows 7 or lower is not guaranteed.

- Download the F5-TTSx .bat installer for Windows in [Releases](https://github.com/LeeAeron/F5TTSx/releases).
- Place the BAT-file in any folder in the root of any partition with a short Latin name without spaces or special characters and run it.
- Select INSTALL (2) entry .bat file will download, unpack and configure all needed environment.
- After installing, select START (1). .bat will launch Browser, and loads necessary files, models, also there will be loaded Official F5-TTS EN/ZH model.


### 🔥 Very important
- F5-TTS_v1 model is an original F5-TTS model used from own repository (https://huggingface.co/LeeAeron/F5TTSx/models/F5TTS_v1_Base)
- F5TTS_RUv2 model is fine-tuned russian model v2 by Misha24-10 used from own repository (https://huggingface.co/LeeAeron/F5TTSx/models/F5TTS_RU/v2)
- F5TTS_RUv4 model is fine-tuned russian model v4 (winter) by Misha24-10 used from own repository (https://huggingface.co/LeeAeron/F5TTSx/models/F5TTS_RU/v4)
- model 

## 📝 License

The **F5-TTSx** code is released under MIT License. 
The pre-trained models are licensed under the CC-BY-NC license due to the training data Emilia, which is an in-the-wild dataset. Sorry for any inconvenience this may cause.


## ⚙️ NVIDIA GPU Support

F5-TTSx supports GTX and RTX cards, including GTX10xx-16xx and RTX 20xx–50xx. Special notes:


## 📥 Additional Models

- [Multilingual (zh & en)](https://huggingface.co/SWivid/F5-TTS/tree/main/F5TTS_v1_Base)
- [Finnish](https://huggingface.co/AsmoKoskinen/F5-TTS_Finnish_Model)
- [French](https://huggingface.co/RASPIAUDIO/F5-French-MixedSpeakers-reduced)
- [German](https://huggingface.co/hvoss-techfak/F5-TTS-German)
- [Hindi](https://huggingface.co/SPRINGLab/F5-Hindi-24KHz)
- [Italian](https://huggingface.co/alien79/F5-TTS-italian)
- [Japanese](https://huggingface.co/Jmica/F5TTS/tree/main/JA_21999120)
- [Latvian](https://huggingface.co/RaivisDejus/F5-TTS-Latvian)
- [Russian](https://huggingface.co/hotstone228/F5-TTS-Russian)
- [Russian](https://huggingface.co/Misha24-10/F5-TTS_RUSSIAN)
- [Spanish](https://huggingface.co/jpgallegoar/F5-Spanish)

## 📺 Credits

* [LeeAeron](https://github.com/LeeAeron) — additional code, modding, reworking, repository, Hugginface space, features, installer/launcher, reference audios, dictionary support.
* [mrfakename](https://github.com/fakerybakery) — original [online demo](https://huggingface.co/spaces/mrfakename/E2-F5-TTS)
* [RootingInLoad](https://github.com/RootingInLoad) — chunk generation & podcast app exploration
* [jpgallegoar](https://github.com/jpgallegoar) — multiple speech-type generation & voice chat

