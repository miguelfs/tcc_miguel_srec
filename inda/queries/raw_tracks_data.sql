select t.session_id,
       s.user_id                                                            as owner_id,
       t.user_id,
       t.created_at,
       case when t.parent_id = '' then 'create_track' else 'fork_track' end as action
from ${project_id}.${db_name}.tracks t
         inner join ${project_id}.${db_name}.sessions s on t.session_id = s.id
order by created_at desc
limit 100
