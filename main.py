"""
FaceOff - AI Face Swapper
Main entry point for the application.
"""
import logging
from ui.app import create_app
from utils.config_manager import config
from utils.model_cache import preload_models, get_cache_info

logger = logging.getLogger("FaceOff")

if __name__ == "__main__":
    # Display model cache info at startup
    cache_info = get_cache_info()
    logger.info(
        "Model cache: %d engine(s) cached (%.2f MB total)",
        cache_info['num_files'],
        cache_info['total_size_mb']
    )

    # Optional: Preload models in background
    if config.preload_on_startup:
        logger.info("Model preloading enabled - compiling TensorRT engines...")
        preload_models(device_id=0)

    # Create Gradio app
    demo = create_app()

    # First attempt: prefer configured port
    try:
        demo.launch(
            server_name=config.server_name,
            server_port=config.server_port,
            share=True  # Force share=True here
        )
    except OSError as e:
        # If main port is busy → try alternative ports
        if "Cannot find empty port" in str(e):
            logger.info(f"Port {config.server_port} is busy, trying alternative ports...")

            for port in range(7861, 7871):
                try:
                    demo.launch(
                        server_name=config.server_name,
                        server_port=port,
                        share=True  # Force share=True also here
                    )
                    logger.info(f"✅ Successfully started on port {port}")
                    break
                except OSError:
                    continue
            else:
                logger.error("❌ Could not find any available port in range 7861–7870")
                raise
        else:
            raise
