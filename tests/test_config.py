import pyaudio

from claude_whisper.config import Config


class TestConfig:
    """Test suite for Config settings loader."""

    def test_default_values(self):
        """Test that default values are loaded correctly."""
        config = Config()

        assert config.model_name == "mlx-community/whisper-medium-mlx-8bit"
        assert config.format == pyaudio.paInt16
        assert config.channels == 1
        assert config.rate == 16000
        assert config.chunk == 1024
        assert config.silence_threshold == 500
        assert config.silence_chunks == 30
        assert config.command == "jarvis"
        assert config.permission_mode == "acceptEdits"
        assert config.push_to_talk_key == "esc"
        assert config.plan_folder == "plans"

    def test_env_var_loading(self, monkeypatch):
        """Test that environment variables with CLAUDE_WHISPER_ prefix are loaded."""
        monkeypatch.setenv("CLAUDE_WHISPER_MODEL_NAME", "custom-model")
        monkeypatch.setenv("CLAUDE_WHISPER_COMMAND", "hey-claude")
        monkeypatch.setenv("CLAUDE_WHISPER_RATE", "22050")
        monkeypatch.setenv("CLAUDE_WHISPER_SILENCE_THRESHOLD", "1000")
        monkeypatch.setenv("CLAUDE_WHISPER_SILENCE_CHUNKS", "50")
        monkeypatch.setenv("CLAUDE_WHISPER_CHUNK", "2048")
        monkeypatch.setenv("CLAUDE_WHISPER_PERMISSION_MODE", "plan")
        monkeypatch.setenv("CLAUDE_WHISPER_PUSH_TO_TALK_KEY", "ctrl+shift+r")
        monkeypatch.setenv("CLAUDE_WHISPER_PLAN_FOLDER", "custom_plans")

        config = Config()

        assert config.model_name == "custom-model"
        assert config.command == "hey-claude"
        assert config.rate == 22050
        assert config.silence_threshold == 1000
        assert config.silence_chunks == 50
        assert config.chunk == 2048
        assert config.permission_mode == "plan"
        assert config.push_to_talk_key == "ctrl+shift+r"
        assert config.plan_folder == "custom_plans"

    def test_toml_file_loading_via_init(self, tmp_path):
        """Test loading TOML file by providing path directly."""
        toml_content = """
model_name = "toml-model"
command = "computer"
rate = 48000
silence_threshold = 750
channels = 2
chunk = 4096
silence_chunks = 60
push_to_talk_key = "ctrl+alt+space"
"""
        toml_file = tmp_path / "test_config.toml"
        toml_file.write_text(toml_content)

        from pydantic_settings import (
            BaseSettings,
            PydanticBaseSettingsSource,
            SettingsConfigDict,
            TomlConfigSettingsSource,
        )

        class TestConfig(BaseSettings):
            model_config = SettingsConfigDict(
                env_prefix="CLAUDE_WHISPER_",
                toml_file=str(toml_file),
                extra="ignore",
            )

            @classmethod
            def settings_customise_sources(
                cls,
                settings_cls: type[BaseSettings],
                init_settings: PydanticBaseSettingsSource,
                env_settings: PydanticBaseSettingsSource,
                dotenv_settings: PydanticBaseSettingsSource,
                file_secret_settings: PydanticBaseSettingsSource,
            ) -> tuple[PydanticBaseSettingsSource, ...]:
                return (
                    init_settings,
                    env_settings,
                    TomlConfigSettingsSource(settings_cls),
                    file_secret_settings,
                )

            model_name: str = "default-model"
            command: str = "jarvis"
            rate: int = 16000
            silence_threshold: int = 500
            channels: int = 1
            chunk: int = 1024
            silence_chunks: int = 30
            push_to_talk_key: str = "esc"

        config = TestConfig()

        assert config.model_name == "toml-model"
        assert config.command == "computer"
        assert config.rate == 48000
        assert config.silence_threshold == 750
        assert config.channels == 2
        assert config.chunk == 4096
        assert config.silence_chunks == 60
        assert config.push_to_talk_key == "ctrl+alt+space"

    def test_priority_init_over_env(self, monkeypatch):
        """Test that init values take priority over environment variables."""
        monkeypatch.setenv("CLAUDE_WHISPER_MODEL_NAME", "env-model")
        monkeypatch.setenv("CLAUDE_WHISPER_COMMAND", "env-command")

        config = Config(model_name="init-model", command="init-command")

        assert config.model_name == "init-model"
        assert config.command == "init-command"

    def test_priority_env_over_defaults(self, monkeypatch):
        """Test that environment variables take priority over defaults."""
        monkeypatch.setenv("CLAUDE_WHISPER_MODEL_NAME", "env-model")

        config = Config()

        assert config.model_name == "env-model"
        # Other fields should still have defaults
        assert config.command == "jarvis"
        assert config.rate == 16000

    def test_push_to_talk_key_variations(self):
        """Test different push-to-talk key combinations."""
        test_cases = ["esc", "ctrl+shift+r", "ctrl+alt+space", "f12", "shift+space"]

        for key in test_cases:
            config = Config(push_to_talk_key=key)
            assert config.push_to_talk_key == key

    def test_multiple_source_priority(self, tmp_path, monkeypatch):
        """Test the priority chain: init > env > toml > defaults."""
        # Setup TOML file
        toml_content = """
model_name = "toml-model"
command = "toml-command"
rate = 48000
silence_threshold = 750
"""
        toml_file = tmp_path / "priority_test.toml"
        toml_file.write_text(toml_content)

        # Setup environment variables
        monkeypatch.setenv("CLAUDE_WHISPER_MODEL_NAME", "env-model")
        monkeypatch.setenv("CLAUDE_WHISPER_RATE", "22050")

        # Create custom config for this test
        from pydantic_settings import (
            BaseSettings,
            PydanticBaseSettingsSource,
            SettingsConfigDict,
            TomlConfigSettingsSource,
        )

        class PriorityTestConfig(BaseSettings):
            model_config = SettingsConfigDict(
                env_prefix="CLAUDE_WHISPER_",
                toml_file=str(toml_file),
                extra="ignore",
            )

            @classmethod
            def settings_customise_sources(
                cls,
                settings_cls: type[BaseSettings],
                init_settings: PydanticBaseSettingsSource,
                env_settings: PydanticBaseSettingsSource,
                dotenv_settings: PydanticBaseSettingsSource,
                file_secret_settings: PydanticBaseSettingsSource,
            ) -> tuple[PydanticBaseSettingsSource, ...]:
                return (
                    init_settings,
                    env_settings,
                    TomlConfigSettingsSource(settings_cls),
                    file_secret_settings,
                )

            model_name: str = "default-model"
            command: str = "default-command"
            rate: int = 16000
            silence_threshold: int = 500
            channels: int = 1

        # Test priority: init > env > toml > default
        config = PriorityTestConfig(model_name="init-model")

        # model_name: init wins
        assert config.model_name == "init-model"
        # rate: env wins (no init value)
        assert config.rate == 22050
        # silence_threshold: toml wins (no init or env)
        assert config.silence_threshold == 750
        # channels: default wins (not in init, env, or toml)
        assert config.channels == 1
