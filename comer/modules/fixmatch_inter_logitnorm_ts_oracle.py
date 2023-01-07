from comer.modules import CoMERFixMatchOracleInterleaved, \
    CoMERFixMatchInterleavedTemperatureScaling


class CoMERFixMatchOracleInterleavedLogitNormTempScale(
    CoMERFixMatchInterleavedTemperatureScaling,
    CoMERFixMatchOracleInterleaved
):
    pass
