from comer.modules import CoMERFixMatchInterleavedFixedPctTemperatureScaling, CoMERFixMatchInterleavedLogitNormTempScale


class CoMERFixMatchInterleavedFixedPctLogitNormTempScale(
    CoMERFixMatchInterleavedLogitNormTempScale,
    CoMERFixMatchInterleavedFixedPctTemperatureScaling
):
    pass
