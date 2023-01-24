from comer.modules import CoMERFixMatchOracleInterleaved, \
    CoMERFixMatchInterleavedTemperatureScaling


class CoMERFixMatchOracleInterleavedTempScale(
    CoMERFixMatchInterleavedTemperatureScaling,
    CoMERFixMatchOracleInterleaved
):
    pass
