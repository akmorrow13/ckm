# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: features.proto

from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import descriptor_pb2
# @@protoc_insertion_point(imports)




DESCRIPTOR = _descriptor.FileDescriptor(
  name='features.proto',
  package='features',
  serialized_pb='\n\x0e\x66\x65\x61tures.proto\x12\x08\x66\x65\x61tures\"M\n\x05\x44\x61tum\x12\x10\n\x04\x64\x61ta\x18\x01 \x03(\x01\x42\x02\x10\x01\x12\r\n\x05label\x18\x02 \x02(\x05\x12\x15\n\rlabel_weights\x18\x03 \x03(\x01\x12\x0c\n\x04name\x18\x04 \x01(\t\"6\n\x07\x44\x61taset\x12\x0c\n\x04name\x18\x03 \x02(\t\x12\x1d\n\x04\x64\x61ta\x18\x04 \x03(\x0b\x32\x0f.features.Datum')




_DATUM = _descriptor.Descriptor(
  name='Datum',
  full_name='features.Datum',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='data', full_name='features.Datum.data', index=0,
      number=1, type=1, cpp_type=5, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=_descriptor._ParseOptions(descriptor_pb2.FieldOptions(), '\020\001')),
    _descriptor.FieldDescriptor(
      name='label', full_name='features.Datum.label', index=1,
      number=2, type=5, cpp_type=1, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='label_weights', full_name='features.Datum.label_weights', index=2,
      number=3, type=1, cpp_type=5, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='name', full_name='features.Datum.name', index=3,
      number=4, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=unicode("", "utf-8"),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  extension_ranges=[],
  serialized_start=28,
  serialized_end=105,
)


_DATASET = _descriptor.Descriptor(
  name='Dataset',
  full_name='features.Dataset',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='name', full_name='features.Dataset.name', index=0,
      number=3, type=9, cpp_type=9, label=2,
      has_default_value=False, default_value=unicode("", "utf-8"),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='data', full_name='features.Dataset.data', index=1,
      number=4, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  extension_ranges=[],
  serialized_start=107,
  serialized_end=161,
)

_DATASET.fields_by_name['data'].message_type = _DATUM
DESCRIPTOR.message_types_by_name['Datum'] = _DATUM
DESCRIPTOR.message_types_by_name['Dataset'] = _DATASET

class Datum(_message.Message):
  __metaclass__ = _reflection.GeneratedProtocolMessageType
  DESCRIPTOR = _DATUM

  # @@protoc_insertion_point(class_scope:features.Datum)

class Dataset(_message.Message):
  __metaclass__ = _reflection.GeneratedProtocolMessageType
  DESCRIPTOR = _DATASET

  # @@protoc_insertion_point(class_scope:features.Dataset)


_DATUM.fields_by_name['data'].has_options = True
_DATUM.fields_by_name['data']._options = _descriptor._ParseOptions(descriptor_pb2.FieldOptions(), '\020\001')
# @@protoc_insertion_point(module_scope)
