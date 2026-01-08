# Code Generation Tooling

## Overview

Code generation enables users to create `PackableSample` classes from schema records published on ATProto, making datasets truly interoperable across different codebases and even languages.

## Goals

1. **Automatic class generation**: Convert schema records to Python classes
2. **Type safety**: Generate proper type hints and validation
3. **Maintainability**: Generated code should be readable and maintainable
4. **Cross-language support** (future): TypeScript, Rust, etc.

## Python Code Generation

### Input: Schema Record

```json
{
  "$type": "app.bsky.atdata.schema",
  "name": "ImageSample",
  "version": "1.0.0",
  "description": "Sample containing an image with label",
  "fields": [
    {
      "name": "image",
      "type": { "kind": "ndarray", "dtype": "uint8", "shape": [null, null, 3] },
      "description": "RGB image with variable height/width"
    },
    {
      "name": "label",
      "type": { "kind": "primitive", "primitive": "str" },
      "description": "Human-readable label"
    },
    {
      "name": "confidence",
      "type": { "kind": "primitive", "primitive": "float" },
      "optional": true,
      "description": "Optional confidence score"
    }
  ]
}
```

### Output: Python Code

```python
"""
ImageSample

Sample containing an image with label

Schema Version: 1.0.0
Schema URI: at://did:plc:abc123/app.bsky.atdata.schema/3jk2lo34klm
Generated: 2025-01-06T12:00:00Z
"""

from dataclasses import dataclass
from typing import Optional
from numpy.typing import NDArray
import atdata


@atdata.packable
class ImageSample:
    """Sample containing an image with label"""

    #: RGB image with variable height/width
    image: NDArray  # uint8, shape: [*, *, 3]

    #: Human-readable label
    label: str

    #: Optional confidence score
    confidence: Optional[float] = None
```

## Code Generator Architecture

### Module Structure

```
src/atdata/codegen/
  __init__.py          # Public API
  generator.py         # Core code generation logic
  templates/           # Template files
    python.jinja2      # Python class template
  cli.py               # CLI interface
  _validators.py       # Schema validation
```

### Core Generator

**File**: `src/atdata/codegen/generator.py`

```python
from typing import Optional
from datetime import datetime, timezone
from jinja2 import Environment, PackageLoader
import atdata
from ..atproto import ATProtoClient, SchemaLoader

class PythonGenerator:
    """Generate Python PackableSample classes from schema records."""

    def __init__(self):
        # Set up Jinja2 environment
        self.env = Environment(
            loader=PackageLoader('atdata.codegen', 'templates'),
            trim_blocks=True,
            lstrip_blocks=True
        )

        # Register custom filters
        self.env.filters['python_type'] = self._python_type_filter
        self.env.filters['python_default'] = self._python_default_filter

    def generate_from_uri(
        self,
        client: ATProtoClient,
        schema_uri: str,
        output_path: Optional[str] = None
    ) -> str:
        """
        Generate Python code from a schema URI.

        Args:
            client: ATProto client
            schema_uri: URI of the schema record
            output_path: Optional path to write output file

        Returns:
            Generated Python code as string
        """
        # Load schema record
        loader = SchemaLoader(client)
        schema = loader.get_schema(schema_uri)

        # Generate code
        code = self.generate_from_record(schema, schema_uri)

        # Write to file if requested
        if output_path:
            with open(output_path, 'w') as f:
                f.write(code)

        return code

    def generate_from_record(
        self,
        schema: dict,
        schema_uri: str
    ) -> str:
        """
        Generate Python code from a schema record dict.

        Args:
            schema: Schema record dict
            schema_uri: URI of the schema (for documentation)

        Returns:
            Generated Python code
        """
        # Validate schema
        self._validate_schema(schema)

        # Prepare template context
        context = {
            'schema': schema,
            'schema_uri': schema_uri,
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'fields': self._prepare_fields(schema['fields'])
        }

        # Render template
        template = self.env.get_template('python.jinja2')
        code = template.render(**context)

        return code

    def _prepare_fields(self, fields: list[dict]) -> list[dict]:
        """Prepare fields for template rendering."""
        prepared = []

        for field in fields:
            prepared.append({
                'name': field['name'],
                'type_annotation': self._field_type_to_python(field['type']),
                'optional': field.get('optional', False),
                'description': field.get('description', ''),
                'type_comment': self._type_comment(field['type'])
            })

        return prepared

    def _field_type_to_python(self, field_type: dict) -> str:
        """Convert schema field type to Python type annotation."""
        kind = field_type['kind']

        if kind == 'primitive':
            primitive_map = {
                'str': 'str',
                'int': 'int',
                'float': 'float',
                'bool': 'bool',
                'bytes': 'bytes'
            }
            return primitive_map[field_type['primitive']]

        elif kind == 'ndarray':
            return 'NDArray'

        elif kind == 'nested':
            # Extract class name from schema ref
            # For now, just use a placeholder
            ref = field_type['schemaRef']
            return f'NestedType'  # TODO: resolve nested types

        else:
            raise ValueError(f"Unknown field type kind: {kind}")

    def _type_comment(self, field_type: dict) -> Optional[str]:
        """Generate type comment for NDArray types."""
        if field_type['kind'] == 'ndarray':
            dtype = field_type['dtype']
            shape = field_type.get('shape')
            if shape:
                shape_str = ', '.join('*' if s is None else str(s) for s in shape)
                return f"{dtype}, shape: [{shape_str}]"
            else:
                return f"{dtype}"
        return None

    def _python_type_filter(self, field: dict) -> str:
        """Jinja2 filter to get Python type annotation."""
        type_str = self._field_type_to_python(field['type'])
        if field.get('optional'):
            return f'Optional[{type_str}]'
        return type_str

    def _python_default_filter(self, field: dict) -> Optional[str]:
        """Jinja2 filter to get Python default value."""
        if field.get('optional'):
            return 'None'
        return None

    def _validate_schema(self, schema: dict) -> None:
        """Validate schema record structure."""
        required = ['name', 'version', 'fields']
        for field in required:
            if field not in schema:
                raise ValueError(f"Schema missing required field: {field}")

        if not isinstance(schema['fields'], list):
            raise ValueError("Schema fields must be a list")

        for field in schema['fields']:
            if 'name' not in field or 'type' not in field:
                raise ValueError(f"Field missing name or type: {field}")
```

### Template File

**File**: `src/atdata/codegen/templates/python.jinja2`

```jinja2
"""
{{ schema.name }}

{{ schema.description }}

Schema Version: {{ schema.version }}
Schema URI: {{ schema_uri }}
Generated: {{ generated_at }}

⚠️  This file was automatically generated from an ATProto schema record.
   Do not edit manually - regenerate using `atdata codegen` instead.
"""

from dataclasses import dataclass
{%- if fields | selectattr('optional') | list %}
from typing import Optional
{%- endif %}
{%- if fields | selectattr('type.kind', 'equalto', 'ndarray') | list %}
from numpy.typing import NDArray
{%- endif %}
import atdata


@atdata.packable
class {{ schema.name }}:
    """{{ schema.description }}"""

{% for field in fields %}
    {%- if field.description %}
    #: {{ field.description }}
    {%- endif %}
    {{ field.name }}: {{ field | python_type }}
    {%- if field.type_comment %} # {{ field.type_comment }}{% endif %}
    {%- if field | python_default %} = {{ field | python_default }}{% endif %}

{% endfor %}
```

### CLI Interface

**File**: `src/atdata/codegen/cli.py`

```python
import click
from pathlib import Path
from ..atproto import ATProtoClient
from .generator import PythonGenerator


@click.group()
def codegen():
    """Code generation tools for atdata."""
    pass


@codegen.command()
@click.argument('schema_uri')
@click.option('--output', '-o', type=click.Path(), help='Output file path')
@click.option('--handle', '-u', help='ATProto handle for authentication')
@click.option('--password', '-p', help='ATProto password')
@click.option('--language', '-l', default='python', type=click.Choice(['python']), help='Output language')
def generate(schema_uri: str, output: str, handle: str, password: str, language: str):
    """Generate code from a schema URI.

    Example:
        atdata codegen generate at://did:plc:abc123/app.bsky.atdata.schema/3jk2lo34klm -o my_sample.py
    """
    # Initialize client
    client = ATProtoClient()

    # Authenticate if credentials provided
    if handle and password:
        client.login(handle, password)

    # Generate code
    generator = PythonGenerator()

    try:
        code = generator.generate_from_uri(client, schema_uri, output)

        if output:
            click.echo(f"Generated {language} code written to {output}")
        else:
            click.echo(code)

    except Exception as e:
        click.echo(f"Error generating code: {e}", err=True)
        raise click.Abort()


@codegen.command()
@click.argument('schema_uris', nargs=-1, required=True)
@click.option('--output-dir', '-d', type=click.Path(), required=True, help='Output directory')
@click.option('--handle', '-u', help='ATProto handle for authentication')
@click.option('--password', '-p', help='ATProto password')
def batch(schema_uris: tuple, output_dir: str, handle: str, password: str):
    """Generate code for multiple schemas.

    Example:
        atdata codegen batch schema1_uri schema2_uri -d ./generated
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Initialize client
    client = ATProtoClient()
    if handle and password:
        client.login(handle, password)

    # Generate code for each schema
    generator = PythonGenerator()

    for schema_uri in schema_uris:
        try:
            # Load schema to get name
            from ..atproto import SchemaLoader
            loader = SchemaLoader(client)
            schema = loader.get_schema(schema_uri)

            # Generate output path from schema name
            filename = f"{schema['name'].lower()}.py"
            output_file = output_path / filename

            # Generate code
            generator.generate_from_uri(client, schema_uri, str(output_file))

            click.echo(f"Generated {filename}")

        except Exception as e:
            click.echo(f"Error generating code for {schema_uri}: {e}", err=True)


if __name__ == '__main__':
    codegen()
```

### Integration with Main CLI

**File**: `src/atdata/cli.py` (new or extend existing)

```python
import click
from .codegen.cli import codegen as codegen_group

@click.group()
def main():
    """atdata command-line interface."""
    pass

# Add codegen subcommand
main.add_command(codegen_group)

if __name__ == '__main__':
    main()
```

**Update** `pyproject.toml`:

```toml
[project.scripts]
atdata = "atdata.cli:main"
```

## Usage Examples

### Generate Single Schema

```bash
# Generate Python code from schema URI
atdata codegen generate \
  at://did:plc:abc123/app.bsky.atdata.schema/3jk2lo34klm \
  -o image_sample.py

# Output to stdout instead
atdata codegen generate \
  at://did:plc:abc123/app.bsky.atdata.schema/3jk2lo34klm
```

### Batch Generation

```bash
# Generate multiple schemas to a directory
atdata codegen batch \
  at://did:plc:abc123/app.bsky.atdata.schema/schema1 \
  at://did:plc:abc123/app.bsky.atdata.schema/schema2 \
  at://did:plc:abc123/app.bsky.atdata.schema/schema3 \
  -d ./generated_schemas
```

### Programmatic Usage

```python
from atdata.atproto import ATProtoClient
from atdata.codegen import PythonGenerator

# Initialize
client = ATProtoClient()
client.login("alice.bsky.social", "password")

# Generate code
generator = PythonGenerator()
code = generator.generate_from_uri(
    client,
    "at://did:plc:abc123/app.bsky.atdata.schema/3jk2lo34klm",
    output_path="my_sample.py"
)

# Now can import and use the generated class
from my_sample import ImageSample

# Use with Dataset
dataset = atdata.Dataset[ImageSample](url="s3://bucket/data-{000000..000009}.tar")
```

## Type Validation

### Schema Compatibility Checking

```python
from atdata.codegen import SchemaValidator

class SchemaValidator:
    """Validate schema compatibility and evolution."""

    def is_compatible(self, old_schema: dict, new_schema: dict) -> tuple[bool, list[str]]:
        """
        Check if new_schema is compatible with old_schema.

        Returns:
            (is_compatible, list_of_incompatibilities)
        """
        incompatibilities = []

        # Check for removed fields
        old_fields = {f['name']: f for f in old_schema['fields']}
        new_fields = {f['name']: f for f in new_schema['fields']}

        for name in old_fields:
            if name not in new_fields:
                incompatibilities.append(f"Field removed: {name}")

        # Check for type changes
        for name in old_fields:
            if name in new_fields:
                old_type = old_fields[name]['type']
                new_type = new_fields[name]['type']
                if old_type != new_type:
                    incompatibilities.append(
                        f"Field type changed: {name} from {old_type} to {new_type}"
                    )

        # Check for optional -> required changes
        for name in old_fields:
            if name in new_fields:
                was_optional = old_fields[name].get('optional', False)
                is_optional = new_fields[name].get('optional', False)
                if was_optional and not is_optional:
                    incompatibilities.append(
                        f"Field changed from optional to required: {name}"
                    )

        return len(incompatibilities) == 0, incompatibilities

    def validate_evolution(self, old_version: str, new_version: str) -> bool:
        """Validate that version numbers follow semantic versioning."""
        # Parse versions
        old_major, old_minor, old_patch = map(int, old_version.split('.'))
        new_major, new_minor, new_patch = map(int, new_version.split('.'))

        # Major version should increment for breaking changes
        # Minor version should increment for compatible additions
        # Patch version should increment for bug fixes

        return new_major >= old_major
```

### Runtime Type Validation

```python
from atdata.codegen import TypeValidator

class TypeValidator:
    """Validate sample instances against schemas."""

    def validate(self, sample: atdata.PackableSample, schema: dict) -> tuple[bool, list[str]]:
        """
        Validate that a sample instance conforms to a schema.

        Returns:
            (is_valid, list_of_errors)
        """
        errors = []

        # Check all required fields present
        schema_fields = {f['name']: f for f in schema['fields']}

        for field_name, field_def in schema_fields.items():
            if not field_def.get('optional', False):
                if not hasattr(sample, field_name):
                    errors.append(f"Missing required field: {field_name}")

        # Check field types
        for field_name, field_def in schema_fields.items():
            if hasattr(sample, field_name):
                value = getattr(sample, field_name)
                if value is not None:
                    type_valid = self._validate_field_type(value, field_def['type'])
                    if not type_valid:
                        errors.append(
                            f"Invalid type for field {field_name}: "
                            f"expected {field_def['type']}, got {type(value)}"
                        )

        return len(errors) == 0, errors

    def _validate_field_type(self, value, field_type: dict) -> bool:
        """Validate that value matches field type."""
        kind = field_type['kind']

        if kind == 'primitive':
            primitive_types = {
                'str': str,
                'int': int,
                'float': float,
                'bool': bool,
                'bytes': bytes
            }
            expected_type = primitive_types[field_type['primitive']]
            return isinstance(value, expected_type)

        elif kind == 'ndarray':
            import numpy as np
            if not isinstance(value, np.ndarray):
                return False

            # Check dtype if specified
            if 'dtype' in field_type:
                expected_dtype = np.dtype(field_type['dtype'])
                if value.dtype != expected_dtype:
                    return False

            # Check shape if specified
            if 'shape' in field_type and field_type['shape']:
                expected_shape = field_type['shape']
                if len(value.shape) != len(expected_shape):
                    return False
                for actual_dim, expected_dim in zip(value.shape, expected_shape):
                    if expected_dim is not None and actual_dim != expected_dim:
                        return False

            return True

        return True
```

## Testing

### Unit Tests

```python
import pytest
from atdata.codegen import PythonGenerator

def test_generate_simple_schema():
    """Test generating code from a simple schema."""
    schema = {
        "name": "TestSample",
        "version": "1.0.0",
        "description": "Test sample",
        "fields": [
            {
                "name": "field1",
                "type": {"kind": "primitive", "primitive": "str"}
            }
        ]
    }

    generator = PythonGenerator()
    code = generator.generate_from_record(schema, "at://test/schema/123")

    # Check that code contains expected elements
    assert "@atdata.packable" in code
    assert "class TestSample:" in code
    assert "field1: str" in code


def test_generate_ndarray_field():
    """Test generating code with NDArray fields."""
    schema = {
        "name": "ImageSample",
        "version": "1.0.0",
        "description": "Image sample",
        "fields": [
            {
                "name": "image",
                "type": {
                    "kind": "ndarray",
                    "dtype": "uint8",
                    "shape": [None, None, 3]
                }
            }
        ]
    }

    generator = PythonGenerator()
    code = generator.generate_from_record(schema, "at://test/schema/456")

    assert "from numpy.typing import NDArray" in code
    assert "image: NDArray" in code
    assert "# uint8, shape: [*, *, 3]" in code


def test_optional_fields():
    """Test generating code with optional fields."""
    schema = {
        "name": "OptionalSample",
        "version": "1.0.0",
        "description": "Sample with optional fields",
        "fields": [
            {
                "name": "required_field",
                "type": {"kind": "primitive", "primitive": "str"}
            },
            {
                "name": "optional_field",
                "type": {"kind": "primitive", "primitive": "int"},
                "optional": True
            }
        ]
    }

    generator = PythonGenerator()
    code = generator.generate_from_record(schema, "at://test/schema/789")

    assert "from typing import Optional" in code
    assert "required_field: str" in code
    assert "optional_field: Optional[int] = None" in code
```

### Integration Tests

```python
def test_generate_and_import():
    """Test that generated code can be imported and used."""
    import tempfile
    import importlib.util

    schema = {
        "name": "GeneratedSample",
        "version": "1.0.0",
        "description": "Generated sample",
        "fields": [
            {"name": "x", "type": {"kind": "primitive", "primitive": "int"}}
        ]
    }

    generator = PythonGenerator()

    # Generate code to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        code = generator.generate_from_record(schema, "at://test/schema/123")
        f.write(code)
        temp_path = f.name

    # Import the generated module
    spec = importlib.util.spec_from_file_location("generated", temp_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Test instantiation
    sample = module.GeneratedSample(x=42)
    assert sample.x == 42

    # Test serialization
    assert isinstance(sample, atdata.PackableSample)
    packed = sample.packed
    assert isinstance(packed, bytes)
```

## Implementation Checklist (Phase 4)

- [ ] Implement `PythonGenerator` core logic
- [ ] Create Jinja2 template for Python classes
- [ ] Add CLI commands (`generate`, `batch`)
- [ ] Implement schema validation
- [ ] Implement type compatibility checking
- [ ] Write unit tests for generator
- [ ] Write integration tests (generate + import)
- [ ] Add documentation and examples
- [ ] Consider edge cases (nested types, complex shapes)

## Future Extensions

### Multi-Language Support

**TypeScript Generator**:
```typescript
// Generated from schema
export interface ImageSample {
  image: number[][][];  // uint8, [*, *, 3]
  label: string;
  confidence?: number;
}
```

**Rust Generator**:
```rust
// Generated from schema
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageSample {
    /// RGB image with variable height/width
    pub image: ndarray::Array3<u8>,
    /// Human-readable label
    pub label: String,
    /// Optional confidence score
    pub confidence: Option<f64>,
}
```

### Advanced Features

- **Backwards compatibility checks**: Ensure schema updates don't break existing code
- **Migration generators**: Generate migration code for schema evolution
- **Validation decorators**: Runtime validation of generated classes
- **Documentation generation**: Generate API docs from schemas
- **IDE support**: Language server protocol support for autocomplete

### Code Quality

- **Formatting**: Run `black` on generated Python code
- **Linting**: Ensure generated code passes `ruff`/`flake8`
- **Type checking**: Ensure generated code passes `mypy`
