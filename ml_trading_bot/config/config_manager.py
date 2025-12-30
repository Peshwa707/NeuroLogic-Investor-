"""
Configuration manager for storing and retrieving API keys and settings
Supports both environment variables and persistent storage
"""
import os
import json
from pathlib import Path
from typing import Optional, Dict
import logging

logger = logging.getLogger(__name__)


class ConfigManager:
    """Manage API keys and configuration settings"""

    def __init__(self, config_file: str = None):
        """
        Initialize configuration manager

        Args:
            config_file: Path to configuration file (default: .config/settings.json)
        """
        if config_file is None:
            # Use a hidden directory in user's home or project root
            config_dir = Path.home() / ".ml_trading_bot"
            config_dir.mkdir(exist_ok=True)
            config_file = config_dir / "settings.json"

        self.config_file = Path(config_file)
        self.config = self._load_config()

    def _load_config(self) -> Dict:
        """Load configuration from file"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading config: {e}")
                return {}
        return {}

    def _save_config(self):
        """Save configuration to file"""
        try:
            self.config_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
            logger.info(f"Configuration saved to {self.config_file}")
        except Exception as e:
            logger.error(f"Error saving config: {e}")

    def get(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """
        Get configuration value

        Priority: Environment variable > Stored config > Default

        Args:
            key: Configuration key
            default: Default value if not found

        Returns:
            Configuration value or default
        """
        # First check environment variable
        env_value = os.getenv(key)
        if env_value:
            return env_value

        # Then check stored config
        if key in self.config:
            return self.config[key]

        return default

    def set(self, key: str, value: str):
        """
        Set configuration value

        Args:
            key: Configuration key
            value: Configuration value
        """
        self.config[key] = value
        self._save_config()

    def set_multiple(self, config_dict: Dict[str, str]):
        """
        Set multiple configuration values

        Args:
            config_dict: Dictionary of key-value pairs
        """
        self.config.update(config_dict)
        self._save_config()

    def delete(self, key: str):
        """
        Delete configuration value

        Args:
            key: Configuration key
        """
        if key in self.config:
            del self.config[key]
            self._save_config()

    def get_all(self) -> Dict:
        """
        Get all configuration values (excluding environment variables)

        Returns:
            Dictionary of all stored config
        """
        return self.config.copy()

    def clear_all(self):
        """Clear all stored configuration"""
        self.config = {}
        self._save_config()

    def is_configured(self) -> bool:
        """
        Check if basic API keys are configured

        Returns:
            True if at least one API key is set
        """
        required_keys = [
            'ALPHA_VANTAGE_API_KEY',
            'BINANCE_API_KEY',
            'NEWS_API_KEY'
        ]

        for key in required_keys:
            if self.get(key):
                return True

        return False

    def get_api_status(self) -> Dict[str, bool]:
        """
        Get status of all API keys

        Returns:
            Dictionary with API key status (True if configured)
        """
        api_keys = {
            'Alpha Vantage': 'ALPHA_VANTAGE_API_KEY',
            'Binance': 'BINANCE_API_KEY',
            'Binance Secret': 'BINANCE_API_SECRET',
            'News API': 'NEWS_API_KEY'
        }

        status = {}
        for name, key in api_keys.items():
            value = self.get(key)
            status[name] = bool(value and value != 'your_api_key_here')

        return status

    def validate_api_key(self, key: str, value: str) -> tuple[bool, str]:
        """
        Validate API key format

        Args:
            key: API key name
            value: API key value

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not value or value.strip() == '':
            return False, "API key cannot be empty"

        if value == 'your_api_key_here':
            return False, "Please enter a valid API key"

        if len(value) < 10:
            return False, "API key seems too short"

        return True, "Valid"


# Global instance
config_manager = ConfigManager()


if __name__ == "__main__":
    # Example usage
    import logging
    logging.basicConfig(level=logging.INFO)

    manager = ConfigManager()

    # Set API keys
    manager.set('ALPHA_VANTAGE_API_KEY', 'test_key_123')
    manager.set('BINANCE_API_KEY', 'test_binance_key')

    # Get API keys
    print(f"Alpha Vantage Key: {manager.get('ALPHA_VANTAGE_API_KEY')}")
    print(f"Binance Key: {manager.get('BINANCE_API_KEY')}")

    # Check status
    print(f"\nAPI Status: {manager.get_api_status()}")
    print(f"Is Configured: {manager.is_configured()}")

    # Get all config
    print(f"\nAll Config: {manager.get_all()}")
