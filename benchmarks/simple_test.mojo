struct TestConfig:
    var value: Int

    def __init__(
        out self,
        val: Int = 10,
    ):
        self.value = val


def main():
    var config = TestConfig(5)
    print("Value:", config.value)
