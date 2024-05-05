# from peewee import Model, UUIDField, DateTimeField, ForeignKeyField
# from datetime import datetime
# from uuid import uuid4

# from database.config import DATABASE_TYPE
# from database.connection.postgres import get_postgres_database
# from database.connection.sqlite import get_sqlite_database


# # Establish connection to the database
# if DATABASE_TYPE == "postgres":
#     database = get_postgres_database()
# else:
#     database = get_sqlite_database()


# class BaseModel(Model):
#     id = UUIDField(primary_key=True, default=uuid4)
#     created_at = DateTimeField(default=datetime.now)
#     updated_at = DateTimeField(default=datetime.now)

#     class Meta:
#         database = database


# class RollbackHistory(BaseModel):
#     app = ForeignKeyField(App, on_delete='CASCADE')
#     step_id = UUIDField()
#     timestamp = DateTimeField(default=datetime.now)
#     step_data = JSONField()

#----------------------2nd------------------------------

# from peewee import Model, UUIDField, DateTimeField, ForeignKeyField
# from datetime import datetime
# from uuid import uuid4
# from database.config import DATABASE_TYPE
# from database.connection.postgres import get_postgres_database
# from database.connection.sqlite import get_sqlite_database

# # Establish connection to the database
# if DATABASE_TYPE == "postgres":
#     database = get_postgres_database()
# else:
#     database = get_sqlite_database()

# class BaseModel(Model):
#     id = UUIDField(primary_key=True, default=uuid4)
#     created_at = DateTimeField(default=datetime.now)
#     updated_at = DateTimeField(default=datetime.now)

#     class Meta:
#         database = database

# class RollbackHistory(BaseModel):
#     app = ForeignKeyField(App, on_delete='CASCADE')
#     step_id = UUIDField()
#     timestamp = DateTimeField(default=datetime.now)
#     step_data = JSONField()

#---------------------3rd-------------------------------

# from peewee import ForeignKeyField, AutoField, TextField, IntegerField, CharField
# from database.config import DATABASE_TYPE
# from database.models.components.base_models import BaseModel
# from database.models.app import App
# from database.models.components.sqlite_middlewares import JSONField
# from playhouse.postgres_ext import BinaryJSONField


# class DevelopmentSteps(BaseModel):
#     id = AutoField()  # This will serve as the primary key
#     app = ForeignKeyField(App, on_delete='CASCADE')
#     prompt_path = TextField(null=True)
#     llm_req_num = IntegerField(null=True)
#     token_limit_exception_raised = TextField(null=True)

#     if DATABASE_TYPE == 'postgres':
#         messages = BinaryJSONField(null=True)
#         llm_response = BinaryJSONField(null=False)
#         prompt_data = BinaryJSONField(null=True)
#     else:
#         messages = JSONField(null=True)  # Custom JSON field for SQLite
#         llm_response = JSONField(null=False)  # Custom JSON field for SQLite
#         prompt_data = JSONField(null=True)

#     previous_step = ForeignKeyField('self', null=True, column_name='previous_step')
#     high_level_step = CharField(null=True)

#     class Meta:
#         table_name = 'development_steps'
#         indexes = (
#             (('app', 'previous_step', 'high_level_step'), True),
#         )


#-----------------------4th-------------------------------


from peewee import Model, UUIDField, DateTimeField
from datetime import datetime
from uuid import uuid4

from database.config import DATABASE_TYPE
from database.connection.postgres import get_postgres_database
from database.connection.sqlite import get_sqlite_database


# Establish connection to the database
if DATABASE_TYPE == "postgres":
    database = get_postgres_database()
else:
    database = get_sqlite_database()


class BaseModel(Model):
    id = UUIDField(primary_key=True, default=uuid4)
    created_at = DateTimeField(default=datetime.now)
    updated_at = DateTimeField(default=datetime.now)

    class Meta:
        database = database