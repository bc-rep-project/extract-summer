from datetime import datetime
from peewee import ForeignKeyField, DateTimeField, TextField
from database.models.components.base_models import BaseModel
from database.models.app import App
from database.models.development_steps import DevelopmentSteps

class Checkpoint(BaseModel):
    app = ForeignKeyField(App, on_delete='CASCADE')
    timestamp = DateTimeField(default=datetime.now)
    description = TextField(null=True)
    last_development_step = ForeignKeyField(DevelopmentSteps, column_name='last_development_step')

    class Meta:
        table_name = 'checkpoint'