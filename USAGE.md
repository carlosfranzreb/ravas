# VERANDA Audio/Video Stream Processing

# Prerequisites

> **NOTE:** For minimal testing of the anonymization functionality, the new default avatar renderer as __NO__ additional
>           dependencies. The following minimal requirements only concern the __BROWSER__ based avatar renderer.
>           \
>           _(the avatar renderer can be changed in the `Advanced Settings` tab of the configuration)_
>           \
>           For minimal testing of the anonymization functionality with the __BROWSER__ based avatar renderer, 
>           the [Chrome Browser][1] is _only_ required for the `"Avatar"` video anonymization; however, for the 
>           `"FaceMask"` video anonymization, __no__ other software is required!
>           \
>           For minimal testing, you should disable "Enable Virtual Camera Output:" in the `Configuration/Settings` tab, and instead enable
>           "Show Video Output Window (DEBUG):" in the `Configuration/Advanced Settings` tab.  
>           \
>           This will allow to test the anonymization functionality for video (and for audio). Additional software (see below) is necessary
>           to make the anonymized audio and video available in other communication software, e.g. to select them as virtual camera 
>           or microphone ("audio loopback") in video conferencing software.


 * [Chrome Browser][1] required for rendering avatar with the __browser-based__ avatar renderer
   _(NOTE the rendering is done locally, even in case of using the Chrome browser as rendering engine; in this case, 
     rendering uses the local port `8888` by default to communicate with the browser locally)

 * Virtual Camera: for using anonymized camera output as virtual camera
   * `Windows`: either of these 2 solutions
     * [Open Broadcaster Software][2] (OBS) an Open Source project for streaming
     * [Unity Video Capture][3]
   * `MacOS`:
     * [Open Broadcaster Software][2] (OBS) an Open Source project for streaming
   * `Linux`:
     * the `v4l2loopback` package
     
 * Audio Loopback solution: for using anonymized audio output as virtual microphone
   * `Windows`: _still working on good solution, for now:_
     * [VB-CABLE Driver][4]   
       __NOTE__ the use of the _"New Package"_ version >= `45` is recommended, as the _"Old Package"_ versions (<= `43`) will show the microphone permanently open for `Windows 10` / `11`
   * `MacOS`:
     * the `portaudio` and `blackhole-2ch` package, install with [brew][5]:
       ```bash
       brew install portaudio
       brew install blackhole-2ch
       ```
   * `Linux`:
     * the `portaudio19` and `???` package


# Usage

**T.B.D.**

## Configuration


### Audio Output Configuration

In order to use the (anonymized) audio output in other applications, you should select your microphone in `Audio Input:` (in tab `Settings`), and also
select the _output device_ for your virtual microphone / audio loopback solution in the setting for the `Audio Output:` (in tab `Settings`).

 * Example `VB-CABLE Driver` (`Windows` only):  
   In the VERANDA Streamer configuration, select `CABLE Input (VB-Audio Virtual Cable)` or `CABLE In 16ch (VB-Audio Virtual Cable)` or similiar (the end of the label might get truncated)
   in the `Audio Input:` settting (in tab `Settings`).
   In your other application (e.g. Zoom / Video Conferencing software), open the configuration and select the microphone `CABLE Output (VB-Audio Virtual Cable)` 
   or similiar (the end of the label might get truncated).
   * Prerequisites:
     * install [VB-CABLE Driver][4]
     * in VERANDA Streamer configuration, select your microphone in `Audio Input:` (in tab `Settings`)


### Video Output Configuration

In order to use the (anonymized) video output in other applications, you should enable the option `Enable Virtual Camera Output:` (in tab `Settings`) 
and then select the virtual camera in your other application.

 * Example `OBS Virtual Camera`:  
   In your other application (e.g. Zoom / Video Conferencing software), open the configuration and select the camera `OBS Virtual Camera`. 
   When the VERANDA Streamer is not running, it will show a default still picture. When the VERANDA Streamer is running and streaming, it will show the (anonymized) video output.
   * Prerequisites:
     * install [Open Broadcaster Software][2] (OBS)
     * in VERANDA Streamer configuration, select your camera in `Video Input:` (in tab `Settings`; you may need to press the `Detect` button first)



### Advanced Settings

 * `Advanced Settings`:
   * `Show Video Output Window (DEBUG):` enabled / disable a debug video output window that shows the anonymized video output
   * `Avatar Renderer:` select the rendering engine for avatar-anonyminizer:  \
                        either `OpenGL` for internal renderer, or `Browser` for browser-based renderer (see notes above w.r.t. to
                        browser-based renderer; requires installed `Google Chrome` browser)
   * `Show Avatar Renderer Window (DEBUG):` enabled / disable the window for the avatar renderer (for DEBUG purposes).  \
                                            __NOTE__ this is a _different_ window than the `Video Output Window` 
                                            (the later one shows the final result/camera image).



[1]: https://www.google.com/chrome/
[2]: https://obsproject.com/
[3]: https://unity.com/
[4]: https://vb-audio.com/Cable/
[5]: https://brew.sh/
[6]: https://obsproject.com/kb/virtual-camera-guide
