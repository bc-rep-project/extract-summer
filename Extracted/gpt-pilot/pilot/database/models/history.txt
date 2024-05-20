from datetime import datetime

from peewee import TextField, DateTimeField, ForeignKeyField

from database.models.components.base_models import BaseModel
from database.models.app import App

class CommandHistory(BaseModel):
    app = ForeignKeyField(App, on_delete='CASCADE')
    command = TextField()
    timestamp = DateTimeField(default=datetime.now)

    class Meta:
        table_name = 'command_history'

class UserInputHistory(BaseModel):
    app = ForeignKeyField(App, on_delete='CASCADE')
    input = TextField()
    timestamp = DateTimeField(default=datetime.now)

    class Meta:
        table_name = 'user_input_history'