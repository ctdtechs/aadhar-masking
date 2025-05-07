from flask import Flask
from flasgger import Swagger
from apscheduler.schedulers.background import BackgroundScheduler
from .routes import api_blueprint
from .config import Config
from .scheduler import scheduled_task

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    # Initialize Swagger
    swagger = Swagger(app, template={
        "swagger": "2.0",
        "info": {
            "title": "Traffic Management API",
            "description": "API documentation for Traffic Management API",
            "version": "1.0.0"
        },
        "host": "127.0.0.1:5000",
        "basePath": "/api",
        "schemes": ["http"]
    })

    # Register Blueprints
    app.register_blueprint(api_blueprint)

    # Initialize and start the scheduler
    scheduler = BackgroundScheduler()
    scheduler.add_job(
        func=scheduled_task,
        trigger="interval",
        seconds=app.config["SCHEDULER_INTERVAL_SECONDS"]
    )
    scheduler.start()

    # Ensure the scheduler shuts down when the app context is closed
    @app.teardown_appcontext
    def shutdown_scheduler(exception=None):
        if scheduler.running:
            scheduler.shutdown(wait=False)

    return app